# Test DNN-HMM acoustic models

Prepare a `$DNN` dir

```commandline
mkdir -p $DNN
cd $DNN
touch path.sh
ln -s $KALDI_SRC/egs/wsj/s5/steps .
ln -s $KALDI_SRC/egs/wsj/s5/utils .
cp ${KALDI_SRC}/egs/librispeech/s5/local/score.sh .
```

Create decoding graph for the regular hybrid models:

```commandline
cd $DNN
for MODEL_ID in \
    tr6   \
    tr7   \
    full  \
    ; do
    MODEL_PATH=$MODEL/hybrid-${MODEL_ID}
    rm -rf $DNN/${MODEL_ID}/graph
    mkdir -p $DNN/${MODEL_ID}/graph
    utils/mkgraph.sh \
        $HMM/local/lang_test \
        ${MODEL_PATH} \
        $DNN/${MODEL_ID}/graph
done
```

Create decoding graph for the monophone hybrid mode:

```commandline
cd $HMM
rm -rf $HMM/local/lang_test_mono_tmp_nosp
rm -rf $HMM/local/lang_test_mono_nosp
./utils/prepare_lang.sh \
    --position-dependent-phones false \
    $HMM/local/dict_test_nosp \
    "<UNK>" \
    $HMM/local/lang_test_mono_tmp_nosp \
    $HMM/local/lang_test_mono_nosp
rm -rf local/lang_test_mono
mkdir -p local/lang_test_mono
cp -R local/lang_test_mono_nosp/* local/lang_test_mono
gunzip -c ${RESOURCE}/3-gram.pruned.1e-7.arpa.gz | \
    arpa2fst --disambig-symbol="#0" --read-symbol-table=local/lang_test_mono/words.txt - local/lang_test_mono/G.fst
utils/validate_lang.pl --skip-determinization-check local/lang_test_mono

cd $DNN
rm -rf $DNN/mono/
mkdir -p $DNN/mono/
utils/mkgraph.sh \
    $HMM/local/lang_test_mono \
    $MODEL/hybrid-mono \
    $DNN/mono/graph
```

Run decoding:

```commandline
cd $DNN
for MODEL_ID in \
    tr6   \
    tr7   \
    full  \
    mono  \
    ; do
    MODEL_PATH=$MODEL/hybrid-${MODEL_ID}
    for DTYPE in test_clean test_other; do
        DECODE_DIR=$DNN/${MODEL_ID}/decode_${DTYPE}
        GRAPH_DIR=$DNN/${MODEL_ID}/graph
        DATA_DIR=$DATA/fbank/$DTYPE
        rm -rf ${DECODE_DIR}
        mkdir -p ${DECODE_DIR}/log
        rm -f jobs
        for JOB in `seq 1 32`; do
            JOB0=$((JOB-1))
            echo "$ACN/fe_am/nnet/infer/infer_am.py" \
                 "--cuda-device ${JOB0}" \
                 "--h5 ${DATA_DIR}/fbank.$JOB0.h5" \
                 "--tran ${DATA_DIR}/tran.$JOB0.jsonl.gz" \
                 "--model-dir ${MODEL_PATH}" \
                 "| latgen-faster-mapped" \
                 "--max-active=7000" \
                 "--beam=13.0" \
                 "--lattice-beam=6.0" \
                 "--acoustic-scale=0.083333" \
                 "--allow-partial=true" \
                 "--word-symbol-table=${GRAPH_DIR}/words.txt" \
                 "${MODEL_PATH}/trans.mdl" \
                 "${GRAPH_DIR}/HCLG.fst ark:-" \
                 "'ark:|gzip -c > ${DECODE_DIR}/lat.$JOB.gz'" \
                 "&> ${DECODE_DIR}/log/decode.$JOB.log" >> jobs
        done
        parallel -j 8 --joblog $LOG/decode.${MODEL_ID}.$DTYPE.txt < jobs
        $DNN/score.sh --cmd "run.pl" ${DATA_DIR} ${GRAPH_DIR} ${DECODE_DIR}
        grep WER ${DECODE_DIR}/wer_* | utils/best_wer.sh &> ${DECODE_DIR}/best_wer.txt
        cat ${DECODE_DIR}/best_wer.txt
    done
done
```

This should show results similar to this. Note that these WERs are from using a (fairly modest) 33MB 3-gram LM with no rescoring:

| DNN-HMM Model | test_clean WER (%) | test_other WER (%) |
|---------------|--------------------|--------------------|
| hybrid-tr6    | 6.4                | 13.2               |
| hybrid-tr7    | 6.1                | 12.5               |
| hybrid-full   | 4.9                | 10.3               |
| hybrid-mono   | 8.6                | 14.8               |
