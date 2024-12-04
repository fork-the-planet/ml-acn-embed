# Train and test HMM acoustic models

## Train HMM acoustic models

Create 6 subsets of `TRA` of increasing size and utterance length:

```commandline
for DTYPE MAX_UTTS MIN_WORDS MAX_WORDS in \
    tr1   2000     4         10        \
    tr2   5000     4         10        \
    tr3   10000    3         15        \
    tr4   70000    1         20        \
    tr5   300000   1         40        \
    tr6   700000   1         80        \
    ; do
    $ACN/fe_am/data/select_data.py \
        --max-utts ${MAX_UTTS} \
        --min-words-per-utt ${MIN_WORDS} \
        --max-words-per-utt ${MAX_WORDS} \
        --dst-tran $DATA/$DTYPE.jsonl.gz \
        --src-tran $DATA/tra.jsonl.gz
done
```

Extract MFCCs:

```commandline
for DTYPE in tr1 tr2 tr3 tr4 tr5 tr6 test_clean test_other cv; do
    rm -rf $DATA/mfcc/$DTYPE
    mkdir -p $DATA/mfcc/$DTYPE
    rm -f jobs
    for NUM in `seq 1 24`; do
        echo "$ACN/fe_am/data/get_kaldi_mfcc.py" \
            "--src-tran $DATA/$DTYPE.jsonl.gz" \
            "--audio-path ${LIBRIHEAVY}/utt_wav" \
            "--dst-path $DATA/mfcc/$DTYPE" \
            "--kaldi-bin ${KALDI_BIN}" \
            "--splits 24" \
            "--num $NUM" >> jobs
    done
    parallel -j 8 < jobs
    cat $DATA/mfcc/$DTYPE/mfcc.scp.* | csort > $DATA/mfcc/$DTYPE/mfcc.scp
    rm $DATA/mfcc/$DTYPE/mfcc.scp.*
    cat $DATA/mfcc/$DTYPE/utt2spk.* | csort > $DATA/mfcc/$DTYPE/utt2spk
    rm $DATA/mfcc/$DTYPE/utt2spk.*
    cat $DATA/mfcc/$DTYPE/text.* | csort > $DATA/mfcc/$DTYPE/text
    rm $DATA/mfcc/$DTYPE/text.*
    cp $DATA/mfcc/$DTYPE/mfcc.scp $DATA/mfcc/$DTYPE/feats.scp
    $KALDI_SRC/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl $DATA/mfcc/$DTYPE/utt2spk > $DATA/mfcc/$DTYPE/spk2utt
done
```

Prepare language resources and other files needed for training:

```commandline
mkdir -p $HMM/local
$ACN/fe_am/hmm/write_dict_nosp.py \
    --lexicon $MISC/forcealign-lexicon.txt \
    --non-sil-phones $ACN/setup/non-sil-phones.json \
    --output-dir $HMM/local/dict_nosp

cd $HMM

touch path.sh
ln -s $KALDI_SRC/egs/wsj/s5/steps .
ln -s $KALDI_SRC/egs/wsj/s5/utils .

./utils/prepare_lang.sh \
    $HMM/local/dict_nosp \
    "<UNK>" \
    $HMM/local/lang_tmp_nosp \
    $HMM/local/lang_nosp

steps/compute_cmvn_stats.sh \
    $DATA/mfcc/tr6 \
    $HMM/exp/make_mfcc/tr6 \
    $DATA/mfcc/tr6

for DTYPE in tr1 tr2 tr3 tr4 tr5 test_clean test_other cv; do
    cp $DATA/mfcc/tr6/cmvn.scp $DATA/mfcc/$DTYPE/
done

for DATANUM in `seq 1 6`; do
    rm -rf $DATA/mfcc/tr${DATANUM}/split20
    utils/split_data.sh --per-utt $DATA/mfcc/tr${DATANUM} 20
    mv $DATA/mfcc/tr${DATANUM}/split20utt $DATA/mfcc/tr${DATANUM}/split20
done
```

Train HMM models of increasing size and complexity:

```commandline
cd $HMM
steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "run.pl" \
  $DATA/mfcc/tr1 $HMM/local/lang_nosp $HMM/exp/model_tr1

for NUM LEAVES PDFS in \
    2   2000   10000 \
    3   2500   15000 \
    4   4200   40000 \
    5   5000   100000 \
    6   7000   150000 \
    ; do

    PREV=$((NUM-1))

    steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "run.pl" \
        $DATA/mfcc/tr${NUM} $HMM/local/lang_nosp $HMM/exp/model_tr${PREV} \
        $HMM/exp/ali_tr${PREV}_tr${NUM}

    steps/train_deltas.sh --boost-silence 1.25 --cmd "run.pl" \
        $LEAVES $PDFS $DATA/mfcc/tr${NUM} $HMM/local/lang_nosp \
        $HMM/exp/ali_tr${PREV}_tr${NUM} $HMM/exp/model_tr${NUM}
done
```

## Test HMM acoustic models

Prepare for testing:

```commandline
cd $HMM
rm -rf $HMM/local/dict_test_nosp $HMM/local/lang_test_nosp $HMM/local/lang_test_tmp_nosp local/lang_test

$ACN/fe_am/hmm/write_dict_nosp.py \
    --lexicon $MISC/testing-lexicon.txt \
    --non-sil-phones $ACN/setup/non-sil-phones.json \
    --output-dir $HMM/local/dict_test_nosp

./utils/prepare_lang.sh \
    $HMM/local/dict_test_nosp \
    "<UNK>" \
    $HMM/local/lang_test_tmp_nosp \
    $HMM/local/lang_test_nosp

mkdir -p local/lang_test
cp -R local/lang_test_nosp/* local/lang_test
gunzip -c $RESOURCE/3-gram.pruned.1e-7.arpa.gz | \
    arpa2fst --disambig-symbol="#0" --read-symbol-table=local/lang_test/words.txt - local/lang_test/G.fst
utils/validate_lang.pl --skip-determinization-check local/lang_test
cp ${KALDI_SRC}/egs/librispeech/s5/local/score.sh local
```

Test HMMs:

```commandline
cd $HMM
for DTYPE in test_clean test_other; do
    rm -rf $DATA/mfcc/$DTYPE/split*
    utils/split_data.sh --per-utt $DATA/mfcc/$DTYPE 10
    mv $DATA/mfcc/$DTYPE/split10utt $DATA/mfcc/$DTYPE/split10
done

for NUM in `seq 1 6`; do
    rm -rf $HMM/exp/model_tr${NUM}/graph
    utils/mkgraph.sh $HMM/local/lang_test \
        $HMM/exp/model_tr${NUM} \
        $HMM/exp/model_tr${NUM}/graph

    for DTYPE in test_clean test_other; do
        DECODE_DIR=$HMM/exp/model_tr${NUM}/decode_$DTYPE
        rm -rf ${DECODE_DIR}
        steps/decode.sh --beam 10 --nj 10 --cmd "run.pl" \
            $HMM/exp/model_tr${NUM}/graph \
            $DATA/mfcc/$DTYPE \
            ${DECODE_DIR}
        grep WER ${DECODE_DIR}/wer_* \
            | utils/best_wer.sh &> \
            ${DECODE_DIR}/best_wer.txt
        cat ${DECODE_DIR}/best_wer.txt
    done
done
```

The resulting WERs should look similar to the following values. We don't care much for the actual values right now; we just want to see the accuracy improving:

| HMM Model | test_clean WER (%) | test_other WER (%) |
|-----------|--------------------|--------------------|
| tr1       | 65.2               | 74.2               |
| tr2       | 44.7               | 57.8               |
| tr3       | 37.0               | 51.2               |
| tr4       | 29.2               | 43.9               |
| tr5       | 25.8               | 40.3               |
| tr6       | 24.3               | 38.7               |

