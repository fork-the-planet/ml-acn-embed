# List of downloadables

## Pretrained models

| File & MD5 Hash | Description |
|-------------------------------------------------------------------------|--------------------|
| [model.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/model.tgz) (705 MB)<br>`d6d788fb75151a43fcfbde2533110bde`| All pretrained models, including DNN-HMM hybrid models (context-dependent and monophone), and audio / phone / grapheme embedders with embedding dimensions 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, and 2048 |

## Extra files required for model training and experiments

| File & MD5 Hash | Description |
|-------------------------------------------------------------------------|--------------------|
| [short-utts.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/short-utts.tgz) (98 MB)<br>`933d42f422bd8ecca5b1fbd5e54beb41`| List of IDs of "short" utterances in Libriheavy that we use for training and experiments. |

## Intermediate files generated during the process

These intermediate files are generated during training and experimenting. Since there is some randomness, we share our files for those who wish to replicate our experiments exactly.

| File & MD5 Hash                                                         | Description                                                                                                        |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [dialect.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/dialect.tgz) (37 KB)<br>`3ddad9b602fc38847e86a547acaf03a4`           | TIMIT word segmentations used for dialect clustering experiment                                                    |
| [dnn.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/dnn.tgz) (1.7 GB)<br>`9c7c35d38493b75201dbd785dfb39c32`              | Testing graphs for DNN-HMM acoustic models                                                                         |
| [hmm.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/hmm.tgz) (121 MB)<br>`1b53262fab28feb1c07b9436393c5d40`              | HMM acoustic models and training+testing graphs                                                                    |
| [kaldi-data-pi-mono.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/kaldi-data-pi-mono.tgz) (39 MB)<br>`f2b61c84f315341ba6e8e5cf4337d4b9`| Intermediate files produced during monophone HMM training                                                          |
| [kaldi-data.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/kaldi-data.tgz) (40 MB)<br>`d341973e2e8abf60d2189ae9d5d429de`        | Intermediate files produced during HMM training                                                                    |
| [oovrecovery.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/oovrecovery.tgz) (12 MB)<br>`484bfd707f588dc6cefb33b74acf6b19`       | Randomly-chosen deficient vocabulary and (erroneous) ASR results for use in OOV recovery experiment                |
| [pronlendist.json.gz](https://ml-site.cdn-apple.com/models/ml-acn-embed/pronlendist.json.gz) (480 B)<br>`ac057a5206ddba4db3cc29e0d0d3ac93`   | Distribution of pronunciation lengths used for embedder training                                                   |
| [wakeword.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/wakeword.tgz) (866 MB)<br>`c3f0625041bce38ff4b81623a186ba1f`         | Files for expected confusion experiment for hypothetical wake-up words                                             |
| [wordclassify.tgz](https://ml-site.cdn-apple.com/models/ml-acn-embed/wordclassify.tgz) (18 MB)<br>`33688385ca427820d7f17a6513b79b8a`      | Randomly-chosen list of inputs (10k words) and lexicons (20k, 100k, 500k, 900k) for word classification experiment |

