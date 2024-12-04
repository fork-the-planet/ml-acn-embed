# Expected confusion for wake-up words

Compute scores for all directly-defined n-grams in the LibriSpeech `3-gram.pruned.1e-7` LM:  

```commandline
LM=3-gram.pruned.1e-7
mkdir -p $EXP/wakeword/
$ACN/embed/pronlendist/get_ngram_scores.py \
    --arpa $RESOURCE/${LM}.arpa.gz \
    --output $EXP/wakeword/str2score.${LM}.pt
```

Compute embeddings for all the ngrams:

```commandline
$ACN/exp/wakeupconfusion/get_embeddings.py \
    --batchsize 5000 \
    --str2score $EXP/wakeword/str2score.${LM}.pt \
    --model $MODEL/embedder-64 \
    --output $EXP/wakeword/embeddings-${LM}.pt
```

Compute the expected confusion for a list of hypothetical wake-up words:

```commandline
$ACN/exp/wakeupconfusion/expected_confusion.py \
    --embeddings $EXP/wakeword/embeddings-${LM}.pt \
    --str2score $EXP/wakeword/str2score.${LM}.pt \
    --model $MODEL/embedder-64 \
    --output $EXP/wakeword/output.${LM}.pt \
    --sigma 0.0413956159860057 &> $EXP/wakeword/log.$LM.txt
```

Plot:

```commandline
$ACN/exp/wakeupconfusion/plot_expected_confusion.py \
    --data $EXP/wakeword/output.${LM}.pt \
    --min-lm-scale=0.85 --max-lm-scale=0.95 \
    --output $EXP/wakeword/wakeup-confusion.$LM.eps
```
