# Train context-dependent DNN-HMM acoustic model

1. Extract fbank feats, assuming 128 GPUs for data-parallel training:
```commandline
for DTYPE SPLITS in \
    tr1               16  \
    tr6               128 \
    tr7               128 \
    tra               128 \
    trb               128 \
    cv                128 \
    exp               128 \
    test_clean        32  \
    test_other        32  \
    ; do
    rm -rf $DATA/fbank/$DTYPE
    mkdir -p $DATA/fbank/$DTYPE
    rm -f jobs
    for NUM in `seq 1 ${SPLITS}`; do
        echo "$ACN/fe_am/data/get_kaldi_fbank_h5.py" \
             "--src-tran $DATA/$DTYPE.jsonl.gz" \
             "--audio-path ${LIBRIHEAVY}/utt_wav" \
             "--dst-path $DATA/fbank/$DTYPE" \
             "--splits ${SPLITS}" \
             "--num ${NUM}" >> jobs
    done
    parallel -j 8 < jobs
    cat $DATA/fbank/$DTYPE/utt2spk.* | csort > $DATA/fbank/$DTYPE/utt2spk
    rm $DATA/fbank/$DTYPE/utt2spk.*
    cat $DATA/fbank/$DTYPE/text.* | csort > $DATA/fbank/$DTYPE/text
    rm $DATA/fbank/$DTYPE/text.*
    $KALDI_SRC/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl $DATA/fbank/$DTYPE/utt2spk > $DATA/fbank/$DTYPE/spk2utt
done
```

2. Get alignments for `tr1`(for debugging), `tr6`, and `cv` data:

```commandline
cd $HMM
for DTYPE in tr1 tr6 cv; do
  rm -rf $DATA/mfcc/$DTYPE/split20; utils/split_data.sh --per-utt $DATA/mfcc/$DTYPE/ 20
  mv $DATA/mfcc/$DTYPE/split20utt $DATA/mfcc/$DTYPE/split20
  rm -rf exp/ali_tr6_$DTYPE
  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "run.pl" \
    $DATA/mfcc/$DTYPE $HMM/local/lang_nosp exp/model_tr6 exp/ali_tr6_${DTYPE}
  $ACN/fe_am/hmm/ali_to_pdf.sh --nj 20 exp/model_tr6 exp/ali_tr6_${DTYPE}
  cd exp/ali_tr6_${DTYPE}
  find . -name "pdf.ark*" -delete
  for FILE in pdf.*.gz; do gunzip -c ${FILE} >> pdf.ark; done; gzip pdf.ark
  cd -
  # Don't bother to actually split; just make exact copies and the trainer will still be happy
  for SPLIT in `seq 0 127`; do
    cp exp/ali_tr6_$DTYPE/pdf.ark.gz $DATA/fbank/$DTYPE/pdf.${SPLIT}.ark.gz
  done
done
```

3. Put together various kaldi files needed for hybrid model and store them in `$WORK/kaldi-data.tgz` and `$WORK/kaldi-data/`

```commandline
copy-transition-model --binary=true $HMM/exp/model_tr6/final.mdl $HMM/exp/model_tr6/trans.mdl
copy-transition-model --binary=false $HMM/exp/model_tr6/final.mdl $HMM/exp/model_tr6/trans.txt
$ACN/fe_am/hmm/get_sil_pdfs.py \
    --trans-model $HMM/exp/model_tr6/trans.txt \
    --sil-phones-int $HMM/local/lang_nosp/phones/silence.int \
    --output $HMM/exp/model_tr6/sil_pdf_ids.txt
tree-info $HMM/exp/model_tr6/tree | grep num-pdfs | cut -d' ' -f2 > $HMM/exp/model_tr6/num_pdfs.txt

tar czf $WORK/kaldi-data.tgz \
  -C $HMM/local/ \
  lang_nosp/ \
  -C $HMM/exp/model_tr6/ \
  sil_pdf_ids.txt \
  trans.mdl \
  trans.txt \
  tree \
  num_pdfs.txt
  
mkdir -p $WORK/kaldi-data
tar xzf $WORK/kaldi-data.tgz -C $WORK/kaldi-data
```

4. Train DNN-HMM acoustic model. For each node, run (with arguments modified appropriately, including environment variables)

```commandline
torchrun \
  --nproc-per-node=${GPUS_PER_NODE} \
  --nnodes=${NUM_NODES} \
  --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} \
  --master-port=${MASTER_PORT} \
  $ACN/fe_am/nnet/train/run_fe_am_trainer.py \
  --acoustic-feat-dim=80 \
  --batch-size-per-node-cv=2 \
  --batch-size-per-node-tr=2 \
  --checkpoint-dir=${CKPT_DIR} \
  --data-loader-workers=4 \
  --data-path-cv=$DATA/fbank/cv \
  --data-path-tr=$DATA/fbank/tr6 \
  --dim-out-fn=$WORK/kaldi-data/num_pdfs.txt \
  --log-steps=1000 \
  --max-epochs=50 \
  --max-lrate-mult=0.01 \
  --model-output-local-dir=${OUTPUT} \
  --model-size=20M \
  --steps-per-epoch=5000 \
  --wait-epochs=3 \
  --warmup-steps-k=10
```

5. Store the new model in `$MODEL/hybrid-tr6`:

```commandline
mkdir -p $MODEL/hybrid-tr6
tar xzf $WORK/kaldi-data.tgz -C $MODEL/hybrid-tr6
cp ${OUTPUT}/model.best.pt $MODEL/hybrid-tr6/am.pt
cp ${OUTPUT}/prior.best.pt $MODEL/hybrid-tr6/prior.pt
rm $MODEL/hybrid-tr6/lang_nosp/L*.fst \
   $MODEL/hybrid-tr6/lang_nosp/words.txt \
   $MODEL/hybrid-tr6/lang_nosp/phones/align_lexicon.*
```

6. Recompute the alignments using the new model and retrain the transition probabilities. Note that in practice, `for` loops like the following should be parallelized using multiple nodes:

```commandline
rm -f /tmp/ali.ark
for RANK in `seq 0 127`; do
  $ACN/fe_am/nnet/train/align.py \
      --ref-tran $DATA/fbank/tr6/tran.$RANK.jsonl.gz \
      --h5 $DATA/fbank/tr6/fbank.$RANK.h5 \
      --lang-dir $WORK/kaldi-data/lang_nosp \
      --model-dir $MODEL/hybrid-tr6 \
      --output-gz /tmp/ali.$RANK.ark.gz \
      --output-type tid
  gunzip -c /tmp/ali.$RANK.ark.gz >> /tmp/ali.ark
done
$KALDI_BIN/train-transitions $MODEL/hybrid-tr6/trans.mdl ark:/tmp/ali.ark $MODEL/hybrid-tr6/trans.mdl
cp $MODEL/hybrid-tr6/trans.mdl $MODEL/hybrid-tr6/final.mdl
copy-transition-model --binary=false $MODEL/hybrid-tr6/trans.mdl $MODEL/hybrid-tr6/trans.txt
```

7. Align the `tr7` & `cv` data using `$MODEL/hybrid-tr6`

```commandline
for DTYPE in cv tr7; do
  mkdir -p $WORK/align/$DTYPE
  for RANK in `seq 0 127`; do
    $ACN/fe_am/nnet/train/align.py \
      --ref-tran $DATA/fbank/$DTYPE/tran.$RANK.jsonl.gz \
      --h5 $DATA/fbank/$DTYPE/fbank.$RANK.h5 \
      --lang-dir $WORK/kaldi-data/lang_nosp \
      --model-dir $MODEL/hybrid-tr6 \
      --output-gz $DATA/fbank/$DTYPE/pdf.$RANK.ark.gz \
      --output-type pid
  done
done
```

8. Train the nnet for `$MODEL/hybrid-tr7`

```commandline
torchrun \
  --nproc-per-node=${GPUS_PER_NODE} \
  --nnodes=${NUM_NODES} \
  --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} \
  --master-port=${MASTER_PORT} \
  $ACN/fe_am/nnet/train/run_fe_am_trainer.py \
  --acoustic-feat-dim=80 \
  --batch-size-per-node-cv=2 \
  --batch-size-per-node-tr=2 \
  --checkpoint-dir=${CKPT_DIR} \
  --data-loader-workers=4 \
  --data-path-cv=$DATA/fbank/cv \
  --data-path-tr=$DATA/fbank/tr7 \
  --dim-out-fn=$WORK/kaldi-data/num_pdfs.txt \
  --log-steps=1000 \
  --max-epochs=50 \
  --max-lrate-mult=0.01 \
  --model-output-local-dir=${OUTPUT} \
  --model-size=30M \
  --steps-per-epoch=10000 \
  --wait-epochs=3 \
  --warmup-steps-k=20
```

9. Repeat Step 5 & 6 on `$MODEL/hybrid-tr7` and the `tr7` dataset to finalize the `$MODEL/hybrid-tr7` model.

10. Repeat Step 7 on the `tra` & `cv` data using `$MODEL/hybrid-tr7`.

11. Repeat Steps 8 & 9 using the `tra` data and model size 50M to train the final hybrid model `$MODEL/hybrid-full`.
