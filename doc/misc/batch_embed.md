# Batch computation of embeddings

You can provide multiple inputs for an embedder for batch inference using `jsonl` ([JSON Lines](https://jsonlines.org/)) files.
The outputs are also written to `jsonl` files.
Note, however, that `jsonl` (or any text format) quickly gets unpractical for storing large numeric arrays.
These tools should be considered a starting point, not sustainable solutions for large-scale tasks.  

Compute audio embeddings for 16 segments of `model/examples/librivox-adrift-in-new-york.wav` specified in `model/examples/audio.jsonl`, and write the embeddings to `audio-emb.jsonl`:
```commandline
acn_embed_audio model/embedder-64 --no-dither \
    --input-jsonl model/examples/audio.jsonl \
    --output-jsonl audio-emb.jsonl
```

Compute text embeddings for 16 phone sequences specified in `model/examples/phones.jsonl` and write the embeddings to `phones-emb.jsonl`:

```commandline
acn_embed_phones model/embedder-64 \
    --input-jsonl model/examples/phones.jsonl \
    --output-jsonl phones-emb.jsonl
```

Compute text embeddings for 16 grapheme sequences specified in `model/examples/graphemes.jsonl` and write the embeddings to `graphemes-emb.jsonl`:

```commandline
acn_embed_graphemes model/embedder-64 \
    --input-jsonl model/examples/graphemes.jsonl \
    --output-jsonl graphemes-emb.jsonl
```
