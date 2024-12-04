# Dialect clustering

Run force-alignment on the `sa1` utterances:

```commandline
mkdir -p $EXP/dialect
$ACN/exp/dialect/get_sa1_tran.py \
    --kaldi-bin ${KALDI_BIN} \
    --kaldi-src ${KALDI_SRC} \
    --lexicon $ACN/exp/dialect/sa1-lexicon.txt \
    --model-path $MODEL/hybrid-full \
    --output $EXP/dialect/sa1.jsonl.gz \
    --timit-path $TIMIT
```
Get audio embeddings for each word in all the `sa1` utterances:

```commandline
for DIM in  \
    16 \
    32 \
    48 \
    64 \
    128 \
    ; do
  $ACN/exp/dialect/get_timit_f_embeddings.py \
    --f-model $MODEL/embedder-$DIM \
    --search-wav sa1.wav \
    --output ${EXP}/dialect/timit-sa1-d${DIM}.pkl \
    --tran $EXP/dialect/sa1.jsonl.gz
done
```

Compute dialect distances, using sigma values from grapheme-based embedding clusters:
```commandline
mkdir -p $EXP/dialect/result
for DIM  SIGMA in  \
    16   0.0777377488913104  \
    32   0.0597072928597669  \
    48   0.0491776134444467  \
    64   0.0413956159860057  \
    128  0.0265971093684479  \
    ; do
    $ACN/exp/dialect/dialect_distance.py \
        --addtree \
        --sigma ${SIGMA} \
        ${EXP}/dialect/timit-sa1-d${DIM}.pkl \
        > $EXP/dialect/result/d${DIM}
done
```

Run ADDTREE to get results as follows:

For d=16:
```
-------------------------------------------------------  New England
|
|                                      ---------  South Midland
|                             ---------|
|                             |        -------------------  Southern
|                             |
------------------------------|                   ----------  Northern
                              |               ----|
                              ----------------|   --------  Western
                                              |
                                              ---  North Midland
```

For d=32:
```
-------------------------------------------------------  New England
|
|                                        -----------  South Midland
|                             -----------|
|                             |          -----------------  Southern
|                             |
------------------------------|                   --------  Northern
                              |              -----|
                              ---------------|    ----------  Western
                                             |
                                             -----  North Midland
```

For d=48:
```
--------------------------------------------------------  New England
|
|                                      ----------  South Midland
|                             ---------|
|                             |        --------------------  Southern
|                             |
------------------------------|                    -------  Northern
                              |                 ---|
                              ------------------|  --------  Western
                                                |
                                                ----  North Midland
```

For d=64:
```
------------------------------------------------------  New England
|
|                                           ---------  South Midland
|                           ----------------|
|                           |               ----------------  Southern
|                           |
----------------------------|                -----------  Northern
                            |                |
                            -----------------|---  North Midland
                                             -|
                                              --------  Western
```

For d=128:
```
---------------------------------------------------  New England
|
|                                   ---------------  South Midland
|                  -----------------|
|                  |                ------------------------  Southern
|                  |
-------------------|                      -----------  Northern
                   |                     -|
                   ----------------------|--------  Western
                                         |
                                         ---  North Midland
```
