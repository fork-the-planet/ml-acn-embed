# Train monophone DNN-HMM acoustic model

Prepare language resources. We set `--position-dependent-phones=false` for this model: 

```commandline
mkdir -p $HMM/local/lang_pi_mono_nosp
cp -R $HMM/local/lang_nosp/* $HMM/local/lang_pi_mono_nosp

cd $HMM
utils/prepare_lang.sh \
  --position-dependent-phones false \
  $HMM/local/dict_nosp \
  "<UNK>" \
  $HMM/local/lang_pi_mono_tmp_nosp \
  $HMM/local/lang_pi_mono_nosp

$ACN/fe_am/hmm/convert_lang_to_1pdf_phones.py \
    $HMM/local/lang_pi_mono_nosp
```

Get a transition model file by training a junk model using some tiny data:

```commandline
steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "run.pl" \
    $DATA/mfcc/tr1 \
    $HMM/local/lang_pi_mono_nosp \
    $HMM/exp/model_pi_mono_tr1
```

Put together various kaldi files needed for the monophone hybrid model and store them in `$HMM/kaldi-data-pi-mono.tgz` and `$WORK/kaldi-data-pi-mono/`

```commandline
copy-transition-model --binary=true \
    $HMM/exp/model_pi_mono_tr1/final.mdl \
    $HMM/exp/model_pi_mono_tr1/trans.mdl
copy-transition-model --binary=false \
    $HMM/exp/model_pi_mono_tr1/final.mdl \
    $HMM/exp/model_pi_mono_tr1/trans.txt
$ACN/fe_am/hmm/get_sil_pdfs.py \
    --trans-model $HMM/exp/model_pi_mono_tr1/trans.txt \
    --sil-phones-int $HMM/local/lang_pi_mono_nosp/phones/silence.int \
    --output $HMM/exp/model_pi_mono_tr1/sil_pdf_ids.txt
tree-info $HMM/exp/model_pi_mono_tr1/tree | grep num-pdfs | cut -d' ' -f2 > $HMM/exp/model_pi_mono_tr1/num_pdfs.txt
tar czf $WORK/kaldi-data-pi-mono.tgz \
  -C $HMM/local/ \
  --transform s/lang_pi_mono_nosp/lang_nosp/ \
  lang_pi_mono_nosp/ \
  -C $HMM/exp/model_pi_mono_tr1/ \
  sil_pdf_ids.txt \
  trans.mdl \
  trans.txt \
  tree \
  num_pdfs.txt

mkdir -p $WORK/kaldi-data-pi-mono
tar xzf $WORK/kaldi-data-pi-mono.tgz -C $WORK/kaldi-data-pi-mono
```

Sanity check (should output the message `"Mapping is identity"`):

```commandline
$ACN/fe_am/hmm/check_pdf_mapping_is_identity.py $WORK/kaldi-data-pi-mono/trans.txt
```

For each node, compute the alignments over the training and cross-validation data:

```commandline
for DTYPE in cv tra; do 
    mkdir -p $WORK/align/$DTYPE
    for RANK in `seq 0 127`; do
        $ACN/fe_am/nnet/train/align.py \
          --ref-tran $DATA/fbank/$DTYPE/tran.${RANK}.jsonl.gz \
          --h5 $DATA/fbank/$DTYPE/fbank.${RANK}.h5 \
          --lang-dir $WORK/kaldi-data/lang_nosp \
          --model-dir $MODEL/hybrid-full \
          --output-gz $DATA/fbank/$DTYPE/pdf.${RANK}.ark.gz \
          --output-type pi_phone0id \
          --pd-table $WORK/kaldi-data/lang_nosp/phones.txt \
          --pi-table $WORK/kaldi-data-pi-mono/lang_nosp/phones.txt
    done
done
```

Train DNN-HMM acoustic model, with arguments and env vars set appropriately:

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
  --data-path-tr=$DATA/fbank/tra \
  --dim-out-fn=$WORK/kaldi-data-pi-mono/num_pdfs.txt \
  --log-steps=1000 \
  --max-epochs=50 \
  --max-lrate-mult=0.01 \
  --model-output-local-dir=$OUTPUT \
  --model-size=50M \
  --steps-per-epoch=10000 \
  --wait-epochs=3 \
  --warmup-steps-k=20
```

Store the new model in `$MODEL/hybrid-mono`:
```commandline
mkdir -p $MODEL/hybrid-mono
tar xzf $WORK/kaldi-data-pi-mono.tgz -C $MODEL/hybrid-mono
cp ${OUTPUT}/model.best.pt $MODEL/hybrid-mono/am.pt
cp ${OUTPUT}/prior.best.pt $MODEL/hybrid-mono/prior.pt
rm $MODEL/hybrid-mono/lang_nosp/L*.fst \
   $MODEL/hybrid-mono/lang_nosp/words.txt \
   $MODEL/hybrid-mono/lang_nosp/phones/align_lexicon.*
```

On each node, recompute the alignments using the new model, then consolidate and retrain the transition model:

```commandline
rm -f /tmp/ali.ark
for RANK in `seq 0 127`; do
  $ACN/fe_am/nnet/train/align.py \
      --ref-tran $DATA/fbank/tra/tran.${RANK}.jsonl.gz \
      --h5 $DATA/fbank/tra/fbank.${RANK}.h5 \
      --lang-dir $WORK/kaldi-data-pi-mono/lang_nosp \
      --model-dir $MODEL/hybrid-mono \
      --output-gz /tmp/ali.${RANK}.ark.gz \
      --output-type tid
  gunzip -c /tmp/ali.${RANK}.ark.gz >> /tmp/ali.ark
done
$KALDI_BIN/train-transitions $MODEL/hybrid-mono/trans.mdl ark:/tmp/ali.ark $MODEL/hybrid-mono/trans.mdl
cp $MODEL/hybrid-mono/trans.mdl $MODEL/hybrid-mono/final.mdl
copy-transition-model --binary=false $MODEL/hybrid-mono/trans.mdl $MODEL/hybrid-mono/trans.txt
```
