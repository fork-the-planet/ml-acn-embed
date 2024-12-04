# Word classification experiment

## Data preparation

Get 10k word samples from `EXP` data:

```commandline
mkdir -p $EXPWC/
$ACN/exp/wordclassify/get_test_words.py \
    --tran $DATA/align/exp/tran.jsonl.gz \
    --min-len-ms 500 \
    --num-words 10000 \
    --output $EXPWC/test_words_tran.jsonl.gz
```

Write the fbank feats and split transcriptions for the test samples

```commandline
mkdir -p $EXPWC/fbank
for NUM in `seq 1 24`; do
    $ACN/fe_am/data/get_kaldi_fbank_h5.py \
        --snip-ref-token \
        --src-tran $EXPWC/test_words_tran.jsonl.gz \
        --audio-path $LIBRIHEAVY/utt_wav \
        --dst-path $EXPWC/fbank \
        --splits 24 \
        --num ${NUM} &
done
wait
cat $EXPWC/fbank/utt2spk.* | csort > $EXPWC/fbank/utt2spk
rm $EXPWC/fbank/utt2spk.*
cat $EXPWC/fbank/text.* | csort > $EXPWC/fbank/text
rm $EXPWC/fbank/text.*
$KALDI_SRC/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl $EXPWC/fbank/utt2spk \
    > $EXPWC/fbank/spk2utt
```

Get word lists:

```commandline
for SIZE SIZE_K in \
    20000   20k   \
    100000  100k  \
    500000  500k  \
    900000  900k  \
    ; do
    $ACN/exp/wordclassify/create_vocab.py \
        --word-list $MISC/all-words.txt \
        --lexicon $MISC/all-lexicon.txt \
        $MISC/forcealign-lexicon.txt \
        $MISC/testing-lexicon.txt \
        --test-words-tran $EXPWC/test_words_tran.jsonl.gz \
        --size ${SIZE} \
        --output $EXPWC/test_lexicon.${SIZE_K}.txt
done
```

The results should look as follows:

| Size (words) | Total prons | Unique prons |
|--------------|-------------|--------------|
| 20k          | 36,458      | 36,031       |
| 100k         | 219,741     | 210,473      |
| 500k         | 1,138,945   | 998,546      |
| 900k         | 2,056,624   | 1,710,209    |


## Baseline classification using DNN acoustic model

Create grammar FST:

```commandline
mkdir -p $EXPWC/baseline
for SIZE    SIZE_K in \
    20000   20k   \
    100000  100k  \
    500000  500k  \
    900000  900k  \
    ; do
    $ACN/exp/wordclassify/create_grammar_fst.py \
        --src-lexicon $EXPWC/test_lexicon.${SIZE_K}.txt \
        --output-fst $EXPWC/baseline/test_grammar.${SIZE_K}.fst \
        --output-words $EXPWC/baseline/test_words.${SIZE_K}.txt
done
```

Create a `dict_nosp` directory:

```commandline
mkdir -p $EXPWC/baseline/
cd $EXPWC/baseline/
touch path.sh
ln -s $KALDI_SRC/egs/wsj/s5/steps .
ln -s $KALDI_SRC/egs/wsj/s5/utils .
for SIZE_K in \
    20k   \
    100k  \
    500k  \
    900k    \
    ; do
    rm -rf $EXPWC/baseline/dict_nosp.${SIZE_K}
    rm -rf $EXPWC/baseline/lang_tmp_nosp.${SIZE_K}
    rm -rf $EXPWC/sbaseline/lang_nosp.${SIZE_K}
    $ACN/fe_am/hmm/write_dict_nosp.py \
        --lexicon $EXPWC/test_lexicon.${SIZE_K}.txt \
        --non-sil-phones $ACN/setup/non-sil-phones.json \
        --output-dir $EXPWC/baseline/dict_nosp.${SIZE_K}
    ./utils/prepare_lang.sh \
        --position-dependent-phones false \
        $EXPWC/baseline/dict_nosp.${SIZE_K} \
        "<UNK>" \
        $EXPWC/baseline/lang_tmp_nosp.${SIZE_K} \
        $EXPWC/baseline/lang_nosp.${SIZE_K}
done
```

Make decoding graphs:

```commandline
cd $EXPWC/baseline/
for SIZE_K in \
    20k   \
    100k  \
    500k  \
    900k  \
    ; do
    rm -rf lang_test.${SIZE_K}
    mkdir -p lang_test.${SIZE_K}
    cp -R lang_nosp.${SIZE_K}/* lang_test.${SIZE_K}
    cp test_grammar.${SIZE_K}.fst lang_test.${SIZE_K}/G.fst
    utils/validate_lang.pl --skip-determinization-check lang_test.${SIZE_K}
    utils/mkgraph.sh \
        --remove-oov \
        lang_test.${SIZE_K} \
        $MODEL/hybrid-mono \
        hybrid-mono.${SIZE_K}/graph
done
```

Run decoding:

```commandline
cd $EXPWC/baseline
cp ${KALDI_SRC}/egs/librispeech/s5/local/score.sh .
for SIZE_K in \
    20k   \
    100k  \
    500k  \
    900k \
    ; do
    DECODE_DIR=$EXPWC/baseline/hybrid-mono.${SIZE_K}/decode
    GRAPH_DIR=$EXPWC/baseline/hybrid-mono.${SIZE_K}/graph
    DATA_DIR=$EXPWC/fbank
    rm -rf ${DECODE_DIR}
    mkdir -p ${DECODE_DIR}/log
    rm -rf jobs
    for JOB in `seq 1 24`; do
        JOB0=$((JOB-1))
        echo "$ACN/fe_am/nnet/infer/infer_am.py" \
            "--cuda-device=${JOB0}" \
            "--h5 ${DATA_DIR}/fbank.${JOB0}.h5" \
            "--tran ${DATA_DIR}/tran.${JOB0}.jsonl.gz" \
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
    $ACN/exp/wordclassify/asr_output_to_tran.py \
        --phones-ark ${DECODE_DIR}/phones.ark.gz \
        --phone-table $MODEL/hybrid-mono/lang_nosp/phones.txt \
        --words-ark ${DECODE_DIR}/words.ark.gz \
        --word-table ${GRAPH_DIR}/words.txt \
        --output $EXPWC/baseline/result.${SIZE_K}.jsonl.gz \
        --tran $EXPWC/test_words_tran.jsonl.gz
done
```

Show scores:

```commandline
for SUBWORD in phone grapheme; do
    for SIZE_K in \
        20k   \
        100k  \
        500k  \
        900k \
        ; do
        $ACN/exp/wordclassify/score_match.py \
            --hyp-key asr_token \
            --subword ${SUBWORD} \
            --lexicon $EXPWC/test_lexicon.${SIZE_K}.txt \
            --ref-tran $EXPWC/test_words_tran.jsonl.gz \
            --hyp-tran $EXPWC/baseline/result.${SIZE_K}.jsonl.gz
    done
done
```

which should look like

| Vocab Size | Phone match accuracy (%) | Grapheme match accuracy (%) |
|------------|--------------------------|-----------------------------|
| 20k        | 69.5                     | 68.3                        |
| 100k       | 61.6                     | 56.6                        |
| 500k       | 52.8                     | 37.2                        |
| 900k       | 49.8                     | 29.0                        |

## Classification using embeddings

```commandline
for SUBWORD in phone grapheme; do
    for DIM in 2 4 8 16 32 48 64 128 256 512 1024 2048; do
        export EMBEDDER_ID=d${DIM}
        rm -rf $EXPWC/${EMBEDDER_ID}/$SUBWORD
        mkdir -p $EXPWC/${EMBEDDER_ID}/$SUBWORD
        for SIZE_K in \
            20k   \
            100k  \
            500k  \
            900k    \
            ; do
        
            $ACN/exp/wordclassify/embed_lexicon.py \
                --model $MODEL/embedder-$DIM \
                --subword ${SUBWORD} \
                --lexicon $EXPWC/test_lexicon.${SIZE_K}.txt \
                --output $EXPWC/${EMBEDDER_ID}/$SUBWORD/search.${SIZE_K}.pkl
        
            mkdir -p $EXPWC/${EMBEDDER_ID}/$SUBWORD/result.${SIZE_K}/
        
            rm -f jobs
            for NUM in `seq 0 23`; do
                echo "$ACN/exp/wordclassify/classify_emb.py" \
                    "--cuda-device $NUM" \
                    "--model $MODEL/embedder-$DIM" \
                    "--subword ${SUBWORD}" \
                    "--test-fbank $EXPWC/fbank/fbank.${NUM}.h5" \
                    "--test-tran $EXPWC/fbank/tran.${NUM}.jsonl.gz" \
                    "--trim-sil-thres -0.5" \
                    "--search $EXPWC/${EMBEDDER_ID}/$SUBWORD/search.${SIZE_K}.pkl" \
                    "--output $EXPWC/${EMBEDDER_ID}/$SUBWORD/result.${SIZE_K}/result.${NUM}.jsonl.gz" \
                    >> jobs
            done
            parallel -j 24 < jobs
        done
    done
done
```

Show scores

```commandline
for SUBWORD in phone grapheme; do
    for DIM in 2 4 8 16 32 48 64 128 256 512 1024 2048; do
        export EMBEDDER_ID=d${DIM}
        echo ${EMBEDDER_ID}
        for SIZE_K in \
            20k   \
            100k  \
            500k  \
            900k    \
            ; do
            $ACN/exp/wordclassify/score_match.py \
                --hyp-key hyp_token \
                --subword ${SUBWORD} \
                --lexicon $EXPWC/test_lexicon.${SIZE_K}.txt \
                --ref-tran $EXPWC/test_words_tran.jsonl.gz \
                --hyp-tran $EXPWC/${EMBEDDER_ID}/$SUBWORD/result.${SIZE_K}/result.*.jsonl.gz
        done
    done
done
```

Phone embedding results:

| Embeddings | 20k  | 100k | 500k | 900k |
|------------|------|------|------|------|
| 2          | 1.4  | 0.3  | 0.1  | 0.1  |
| 4          | 22.0 | 8.8  | 3.0  | 1.9  | 
| 8          | 53.9 | 40.5 | 28.4 | 24.1 |
| 16         | 70.1 | 58.2 | 45.8 | 41.3 |
| 32         | 74.8 | 64.0 | 52.5 | 47.9 |
| 48         | 75.5 | 64.9 | 53.0 | 48.1 |
| 64         | 75.4 | 64.5 | 52.8 | 48.4 |
| 128        | 75.5 | 64.9 | 53.0 | 48.7 |
| 256        | 75.3 | 64.6 | 52.5 | 48.3 |
| 512        | 75.3 | 64.4 | 52.2 | 48.0 |
| 1024       | 75.5 | 64.7 | 52.4 | 48.1 |
| 2048       | 75.6 | 64.6 | 52.5 | 47.9 |

Grapheme embedding results:

| Embeddings | 20k  | 100k | 500k | 900k |
|------------|------|------|------|------|
| 2          | 1.6  | 0.3  | 0.1  | 0.0  |
| 4          | 22.6 | 10.3 | 3.4  | 2.4  |
| 8          | 55.1 | 41.1 | 25.4 | 20.0 |
| 16         | 70.7 | 58.5 | 43.2 | 35.9 |
| 32         | 75.2 | 64.0 | 47.8 | 40.4 |
| 48         | 76.3 | 64.9 | 48.2 | 40.7 |
| 64         | 76.1 | 64.8 | 48.0 | 40.7 |
| 128        | 76.4 | 64.5 | 48.3 | 40.7 |
| 256        | 76.3 | 64.3 | 48.2 | 41.2 |
| 512        | 76.1 | 64.8 | 48.1 | 40.7 |
| 1024       | 76.3 | 64.8 | 48.3 | 40.9 |
| 2048       | 76.2 | 64.4 | 48.1 | 40.6 | 
