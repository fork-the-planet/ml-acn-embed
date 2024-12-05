# Data preparation for embedder training

Force-align the Libriheavy data using the DNN-HMM hybrid AM to obtain word segmentations and pronunciations:

```commandline
for DTYPE in trb cv exp; do \
    mkdir -p $DATA/align/$DTYPE
    for RANK in `seq 0 127`; do \
        $ACN/fe_am/nnet/train/align.py \
            --ref-tran $DATA/fbank/$DTYPE/tran.${RANK}.jsonl.gz \
            --h5 $DATA/fbank/$DTYPE/fbank.${RANK}.h5 \
            --lang-dir $WORK/kaldi-data/lang_nosp \
            --model-dir $MODEL/hybrid-full \
            --output-gz $DATA/align/$DTYPE/phones.${RANK}.ark.gz \
            --output-type pd_phone_len
    done
done
```

Convert to transcription files:

```commandline
for DTYPE in trb cv exp; do \
    for RANK in `seq 0 127`; do \
        $ACN/embed/data/phone_ali_to_tran.py \
            --phones-ark $DATA/align/$DTYPE/phones.${RANK}.ark.gz \
            --phone-table $MODEL/hybrid-full/lang_nosp/phones.txt \
            --output $DATA/align/$DTYPE/tran-with-phones.${RANK}.jsonl \
            --tran $DATA/fbank/$DTYPE/tran.${RANK}.jsonl.gz
    done
    cat $DATA/align/$DTYPE/tran-with-phones.*.jsonl | gzip -c > $DATA/align/$DTYPE/tran.jsonl.gz
done
```

Extract AM feats using the monophone DNN-HMM hybrid AM:

```commandline
for DTYPE in trb cv; do \
    mkdir -p $DATA/amfeat/$DTYPE
    for RANK in `seq 0 127`; do \
        $ACN/embed/data/amfeat/get_amfeat.py \
            --fbank-h5 $DATA/fbank/$DTYPE/fbank.${RANK}.h5 \
            --am-dir $MODEL/hybrid-mono \
            --output-h5 $DATA/amfeat/$DTYPE/amfeat.${RANK}.h5
    done
done
```

Compute joint probabilities of n-grams in the language model:

```commandline
$ACN/embed/pronlendist/get_ngram_scores.py \
    --arpa $RESOURCE/3-gram.pruned.1e-7.arpa.gz \
    --output $MISC/ngramscores.3-gram.pruned.1e-7.pkl
```

Compute the distribution of pronunciation lengths:

```commandline
$ACN/embed/pronlendist/get_pronlen_dist.py \
    --string-fn $MISC/ngramscores.3-gram.pruned.1e-7.pkl \
    --src-lexicon $MISC/testing-lexicon.txt \
    --output $MISC/pronlendist.json
```

Gather AM outputs for utterances

```commandline
for ORIG_DTYPE  MAX_SEGS DTYPE in  \
    cv          100000   cv        \
    trb         0        trb       \
  ; do \
    $ACN/embed/data/microbatch/get_long_segments.py \
        --src-tran $DATA/align/${ORIG_DTYPE}/tran.jsonl.gz \
        --src-h5-base-path $DATA/amfeat/${ORIG_DTYPE} \
        --max-segments ${MAX_SEGS} \
        --output-path $DATA/$DTYPE \
        --align-id force_align
    mv $DATA/$DTYPE/metadata.force_align.pkl.gz $DATA/$DTYPE/metadata.pkl.gz 
done
```

Index all segments

```commandline
for DTYPE in \
    cv       \
    trb      \
    ; do \
    $ACN/embed/data/microbatch/index_overlapping_segments.py \
        --metadata-fn $DATA/$DTYPE/metadata.pkl.gz \
        --max-audio-len-frame 1000 \
        --max-pron-len 30 \
        --output-fn $DATA/$DTYPE/index.pkl.gz
done
```

Prepare data for text embedder training

```commandline
for DTYPE       MAX_SIZE in \
    cv          3000000  \
    trb         20000000 \
; do \
    $ACN/embed/data/microbatch/extract_segments_for_g.py \
        --index-fn $DATA/$DTYPE/index.pkl.gz \
        --metadata-fn $DATA/$DTYPE/metadata.pkl.gz \
        --pron-len-dist $MISC/pronlendist.json \
        --max-segments ${MAX_SIZE} \
        --output-pkl $DATA/$DTYPE/g-segments.pkl.gz \
        --output-stats-json $DATA/$DTYPE/g-segments.stats.json
    $ACN/embed/data/microbatch/split_pklgz.py \
        --src-pklgz $DATA/$DTYPE/g-segments.pkl.gz \
        --splits 8
done
```
