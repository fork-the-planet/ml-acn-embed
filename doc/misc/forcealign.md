# Standalone forced-alignment tool

This tool can be used to obtain word segmentations from audio files.
It requires Kaldi binaries and source files, as well as Sequitur G2P for any words that are not in the lexicon (see [setup](../setup.md) notes).

The following is an example using the files in [model.tgz](../downloads.md).

```commandline
$ACN/tools/force_align.py \
    --g2p $RESOURCE/g2p-model-5 \
    --model-dir $MODEL/hybrid-full/ \
    --lexicon $MODEL/examples/adrift-lexicon.txt \
    --wav $MODEL/examples/librivox-adrift-in-new-york.wav \
    --ref "BEFORE I CAME TO NEW YORK TO GO INTO JOURNALISM I TAUGHT SCHOOL FOR TWO YEARS"

[{"orth": "BEFORE", "pron": "B IH0 F AO1 R", "start_ms": 150, "end_ms": 530},
 {"orth": "I", "pron": "AY1", "start_ms": 530, "end_ms": 600},
 {"orth": "CAME", "pron": "K EY1 M", "start_ms": 600, "end_ms": 830},
 {"orth": "TO", "pron": "T UW1", "start_ms": 830, "end_ms": 950},
 {"orth": "NEW", "pron": "N UW1", "start_ms": 950, "end_ms": 1050},
 {"orth": "YORK", "pron": "Y AO1 R K", "start_ms": 1050, "end_ms": 1410},
 {"orth": "TO", "pron": "T UW1", "start_ms": 1410, "end_ms": 1510},
 {"orth": "GO", "pron": "G OW1", "start_ms": 1510, "end_ms": 1650},
 {"orth": "INTO", "pron": "IH1 N T UW0", "start_ms": 1650, "end_ms": 1820},
 {"orth": "JOURNALISM", "pron": "JH ER1 N AH0 L IH2 Z AH0 M", "start_ms": 1820, "end_ms": 2460},
 {"orth": "I", "pron": "AY1", "start_ms": 2710, "end_ms": 2850},
 {"orth": "TAUGHT", "pron": "T AO1 T", "start_ms": 2850, "end_ms": 3100},
 {"orth": "SCHOOL", "pron": "S K UW1 L", "start_ms": 3100, "end_ms": 3410},
 {"orth": "FOR", "pron": "F ER0", "start_ms": 3410, "end_ms": 3530},
 {"orth": "TWO", "pron": "T UW1", "start_ms": 3530, "end_ms": 3710},
 {"orth": "YEARS", "pron": "Y IH1 R Z", "start_ms": 3710, "end_ms": 4170}]
```
