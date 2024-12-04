# Train audio and text embedders

## Train audio embedders

Train a 16-dimensional audio embedder `embedder-16` (replace 16 with other values for other dimensions)

```commandline
torchrun \
    --nproc-per-node=${GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    $ACN/embed/train/f/run_f_trainer.py \
    --batch-size-per-node-cv 1 \
    --batch-size-per-node-tr 1 \
    --checkpoint-dir=${CKPT_DIR} \
    --data-loader-workers 4 \
    --data-path-cv $DATA/cv \
    --data-path-tr $DATA/trb \
    --decay1-steps-k=90 \
    --decay2-factor-k=30 \
    --dim-out=16 \
    --dim-state=100 \
    --dropout-prob=0.4 \
    --log-steps=1000 \
    --max-epochs=100 \
    --max-lrate=0.001 \
    --mbatch-max-same=10 \
    --mbatch-size=160 \
    --model-output-local-dir=${OUTPUT} \
    --num-layers=2 \
    --num-phones-to-prior=$MISC/pronlendist.json \
    --pivots-k-cv 10 \
    --pivots-k-tr 300 \
    --steps-per-epoch=10000 \
    --wait-epochs=10 \
    --warmup-steps-k=30

mkdir -p $MODEL/embedder-16    
cp $OUTPUT/model.best.pt $MODEL/embedder-16/audio.pt
```

Compute embedding clusters to estimate variance, isotropy, etc.

```commandline
for SUBWORD in phone grapheme; do
    rm -rf /mnt/fdata/${SUBWORD}
    mkdir -p /mnt/fdata/${SUBWORD}
    torchrun \
        --nproc-per-node=${GPUS_PER_NODE} \
        --nnodes=${NUM_NODES} \
        --node-rank=${NODE_RANK} \
        --master-addr=${MASTER_ADDR} \
        --master-port=${MASTER_PORT} \
        $ACN/embed/train/f/get_f_clusters.py \
        --data-path $DATA/trb \
        --embedder $MODEL/embedder-16 \
        --min-count 2000 \
        --subword ${SUBWORD} \
        --output-dir /mnt/fdata/${SUBWORD}
done
```

Gather all the files in `/mnt/fdata` in one place and compute stats:

```commandline
for SUBWORD in phone grapheme; do
    $ACN/embed/train/f/get_cluster_stats.py \
        --data /mnt/fdata/${SUBWORD}/f-data.*.pkl \
        --min-samples 2000 \
        --output $MODEL/embedder-16/text-${SUBWORD}-stats.json
done
```

## Train text embedders

Compute audio embeddings over the training data:

```commandline
for DTYPE      in \
    cv  \
    trb \
    ; do
    for RANK in `seq 0 7`; do
        $ACN/embed/train/g/compute_f_output.py \
          --batch-size=1000 \
          --h5=$DATA/$DTYPE/amfeat.h5 \
          --nnet=$MODEL/embedder-16/audio.pt \
          --num-data-workers 4 \
          --output-pkl=$DATA/$DTYPE/foutput.${RANK}.pkl \
          --segments=$DATA/$DTYPE/g-segments.${RANK}.pkl.gz \
          --num-data-workers=4
    done
done
```

Train a phone embedder:

```commandline
torchrun \
    --nproc-per-node=${GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node-rank=${NODE_RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    $ACN/embed/train/g/run_g_trainer.py \
    --batch-size-per-node-cv 8 \
    --batch-size-per-node-tr 8 \
    --checkpoint-dir=${CKPT_DIR} \
    --data-loader-workers 4 \
    --data-path-cv $DATA/cv \
    --data-path-tr $DATA/trb \
    --decay1-steps-k=270 \
    --decay2-factor-k=90 \
    --dim-state=400 \
    --log-steps 5000 \
    --max-epochs 100 \
    --max-lrate 0.001 \
    --model-output-local-dir $OUTPUT \
    --num-layers=1 \
    --steps-per-epoch 20000 \
    --subword phone \
    --subword-json $ACN/setup/non-sil-phones.json \
    --wait-epochs 3 \
    --warmup-steps-k=20

cp $OUTPUT/model.best.pt $MODEL/embedder-16/text-phone.pt
    
```

Above, replace the `--subword` and `--subword-json` arguments to train a grapheme embedder:
```commandline
[...]
    --subword grapheme \
    --subword-json $ACN/embed/model/g_embedder/graphemes.json \
    [...]
    
cp $OUTPUT/model.best.pt $MODEL/embedder-16/text-grapheme.pt
```

