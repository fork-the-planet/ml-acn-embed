# OOV word recovery

Create a deficient vocabulary that excludes the test words:

```commandline
mkdir -p $EXP/oovrecovery
$ACN/exp/oovrecovery/create_deficient_vocab.py \
  --lexicon $MISC/all-lexicon.txt \
  --test-words-tran $EXP/wordclassify/test_words_tran.jsonl.gz \
  --output $EXP/oovrecovery/deficient_lexicon.txt
```

This should result in around 940,010 words, 2,153,255 prons, and 1,793,840 unique prons.

Create grammar, language files, and decoding graph:

```commandline
rm -rf $EXP/oovrecovery/asr
mkdir -p $EXP/oovrecovery/asr

$ACN/exp/wordclassify/create_grammar_fst.py \
  --src-lexicon $EXP/oovrecovery/deficient_lexicon.txt \
  --output-fst $EXP/oovrecovery/asr/deficient_grammar.fst \
  --output-words $EXP/oovrecovery/asr/deficient_words.txt
  
cd $EXP/oovrecovery/asr
ln -s $KALDI_SRC/egs/wsj/s5/steps .
ln -s $KALDI_SRC/egs/wsj/s5/utils .
touch path.sh

$ACN/fe_am/hmm/write_dict_nosp.py \
    --lexicon $EXP/oovrecovery/deficient_lexicon.txt \
    --non-sil-phones $ACN/setup/non-sil-phones.json \
    --output-dir $EXP/oovrecovery/asr/dict_nosp

./utils/prepare_lang.sh \
    --position-dependent-phones false \
    $EXP/oovrecovery/asr/dict_nosp \
    "<UNK>" \
    $EXP/oovrecovery/asr/lang_tmp_nosp \
    $EXP/oovrecovery/asr/lang_nosp

mkdir -p lang_test
cp -R lang_nosp/* lang_test
cp deficient_grammar.fst lang_test/G.fst
utils/validate_lang.pl --skip-determinization-check lang_test

utils/mkgraph.sh \
    --remove-oov \
    lang_test \
    $MODEL/hybrid-mono \
    hybrid-mono/graph
```

Decode using graph with deficient vocabulary:

```commandline
cd $EXP/oovrecovery/asr
cp ${KALDI_SRC}/egs/librispeech/s5/local/score.sh .
DECODE_DIR=$EXP/oovrecovery/asr/hybrid-mono/decode
GRAPH_DIR=$EXP/oovrecovery/asr/hybrid-mono/graph
DATA_DIR=$EXP/wordclassify/fbank
mkdir -p ${DECODE_DIR}/log
rm -rf jobs
for JOB in `seq 1 24`; do
    JOB0=$((JOB-1))
    echo "$ACN/fe_am/nnet/infer/infer_am.py" \
        "--cuda-device ${JOB0}" \
        "--h5 ${DATA_DIR}/fbank.$JOB0.h5" \
        "--tran ${DATA_DIR}/tran.$JOB0.jsonl.gz" \
        "--model-dir $MODEL/hybrid-mono" \
        "|" \
        "decode-faster-mapped" \
        "--max-active=7000" \
        "--beam=80.0" \
        "--acoustic-scale=0.083333" \
        "--allow-partial=false" \
        "--word-symbol-table=${GRAPH_DIR}/words.txt" \
        "$MODEL/hybrid-mono/trans.mdl" \
        "${GRAPH_DIR}/HCLG.fst" \
        "ark:-" \
        "ark,t:${DECODE_DIR}/words.$JOB.ark" \
        "\"ark:| ali-to-phones --write-lengths $MODEL/hybrid-mono/trans.mdl ark:- ark,t:${DECODE_DIR}/phones.$JOB.ark\"" \
        "&> ${DECODE_DIR}/log/decode.$JOB.log" \
        >> jobs
done
parallel -j 24 < jobs
cat ${DECODE_DIR}/phones.*.ark | gzip > ${DECODE_DIR}/phones.ark.gz
cat ${DECODE_DIR}/words.*.ark | gzip > ${DECODE_DIR}/words.ark.gz
```

Gather decoding results:

```commandline
$ACN/exp/wordclassify/asr_output_to_tran.py \
    --phones-ark ${DECODE_DIR}/phones.ark.gz \
    --phone-table $MODEL/hybrid-mono/lang_nosp/phones.txt \
    --words-ark ${DECODE_DIR}/words.ark.gz \
    --word-table ${GRAPH_DIR}/words.txt \
    --output $EXP/oovrecovery/asr/result.jsonl.gz \
    --tran $EXP/wordclassify/test_words_tran.jsonl.gz
```

## OOV recovery using edit distance

Use edit distance between phones for OOV recovery:

```commandline
$ACN/exp/oovrecovery/recover_oov_min_ed.py \
    --model $MODEL/embedder-64 \
    --asr-tran $EXP/oovrecovery/asr/result.jsonl.gz \
    --ref-tran $EXP/wordclassify/test_words_tran.jsonl.gz \
    --output $EXP/oovrecovery/min_ed/oov_min_ed.jsonl.gz
        
$ACN/exp/wordclassify/score_match.py \
    --subword phone \
    --hyp-key hyp_token \
    --ref-tran $EXP/wordclassify/test_words_tran.jsonl.gz \
    --hyp-tran $EXP/oovrecovery/min_ed/oov_min_ed.jsonl.gz
```

## OOV recovery using phone embeddings

Use phone embedder to remap ASR outputs to the OOV vocabulary:

```commandline
for DIM in 16 32 48 64 128 256; do
    $ACN/exp/oovrecovery/embed_oovs.py \
        --model $MODEL/embedder-$DIM \
        --subword phone \
        --ref-tran $EXP/wordclassify/test_words_tran.jsonl.gz \
        --output $EXP/oovrecovery/d${DIM}/oov.pkl

    $ACN/exp/oovrecovery/recover_oov.py \
        --model $MODEL/embedder-$DIM \
        --oovs $EXP/oovrecovery/d${DIM}/oov.pkl \
        --subword phone \
        --asr-tran $EXP/oovrecovery/asr/result.jsonl.gz \
        --output $EXP/oovrecovery/d${DIM}/result.jsonl.gz

    $ACN/exp/wordclassify/score_match.py \
        --subword phone \
        --hyp-key hyp_token \
        --ref-tran $EXP/wordclassify/test_words_tran.jsonl.gz \
        --hyp-tran $EXP/oovrecovery/d${DIM}/result.jsonl.gz
done
```

| Dimensions | Recovery Rate (%) |
|-------------------------|------|
| Baseline (edit distance)| 56.0 |
| 16   | 45.9 |
| 32   | 54.5 |
| 48   | 55.4 |
| 64   | 55.5 |
| 128  | 55.4 |
| 256  | 55.0 |
