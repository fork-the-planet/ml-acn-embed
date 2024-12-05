# Acoustic Neighbor Embeddings

Official source code, documentation, and other files for training models and replicating the experiments in the paper, [_A Theoretical Framework for Acoustic Neighbor Embeddings_](https://arxiv.org/abs/2412.02164).

Pretrained models with an accessible end-user Python interface are also provided.

## Quick start for end users of pretrained embedders

Automatically install the software and essential dependencies: 
```commandline
pip install git+https://github.com/apple/ml-acn-embed@main
```

Download and extract pretrained models (total 705 MB):
```commandline
wget https://ml-site.cdn-apple.com/models/ml-acn-embed/model.tgz -O - | tar xz
```

Compute audio embedding for a segment in an audio file:
```commandline
acn_embed_audio model/embedder-64 --wav model/examples/librivox-adrift-in-new-york.wav \
    --no-dither --start-ms 950 --end-ms 1410
 
[[-0.455  -0.2826 -0.4556 -0.1951  0.0635 -0.2371 -0.1635  0.1702 -0.3562 -0.451   0.2396  0.4862
   0.3348  0.5978 -0.4025 -0.2422 -0.1461 -0.6631 -0.1315 -0.1655  0.5091  0.5982  0.4661 -0.0462
  -1.1064  0.1496  0.321   0.0633  0.3954 -0.0344  0.2964 -0.1347  0.9364  0.3259 -0.6774  0.0106
   0.2444 -0.1617  0.4076  0.0614  0.9511 -0.1825 -0.3518  0.7029  0.0263 -0.0147  0.1475  0.0644
  -0.5739  0.4216 -0.304  -0.1987  0.0066 -0.1506  0.0399 -0.9484 -0.1181 -0.2064 -0.1856 -0.4535
   0.7452  0.1771  0.2255  0.2512]]
```

Compute text embedding for the phone sequence `[N UW1 Y AO1 R K]` (the full list of supported phones is [here](src/acn_embed/setup/non-sil-phones.json), following the [CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)):

```commandline
acn_embed_phones model/embedder-64 --pron "N UW1 Y AO1 R K"

[[-0.4033 -0.312  -0.4508 -0.2395  0.0018 -0.1536 -0.1881  0.1909 -0.3706 -0.408   0.2131  0.4712
   0.3475  0.6508 -0.4154 -0.2339 -0.1123 -0.663  -0.1397 -0.0339  0.5486  0.6744  0.4474 -0.0982
  -1.1024  0.135   0.3345  0.0854  0.4062 -0.0168  0.2736 -0.09    0.9067  0.3249 -0.6679 -0.0083
   0.2754 -0.1247  0.4664  0.139   0.9605 -0.1718 -0.425   0.781   0.0105  0.0487  0.2395  0.0724
  -0.5494  0.3845 -0.3735 -0.196  -0.009  -0.2369  0.0323 -0.9137 -0.0118 -0.1925 -0.2527 -0.3857
   0.8111  0.2302  0.2463  0.2595]]
```

Compute text embedding for the grapheme sequence `NEW YORK` (the space character " " is considered a grapheme; the full list of supported graphemes is [here](src/acn_embed/embed/model/g_embedder/graphemes.json)):

```commandline
acn_embed_graphemes model/embedder-64 --orth "NEW YORK"

[[-4.0224e-01 -3.0963e-01 -4.8059e-01 -2.6145e-01  3.7101e-04 -1.3895e-01 -2.3018e-01  1.8403e-01
  -3.5109e-01 -4.1349e-01  2.4302e-01  4.8039e-01  3.7329e-01  6.2355e-01 -4.2596e-01 -2.1117e-01
  -9.3265e-02 -6.4991e-01 -1.3159e-01 -2.4362e-02  5.3998e-01  6.0438e-01  4.5777e-01 -7.0070e-02
  -1.0676e+00  1.2307e-01  3.2640e-01  8.6234e-02  4.2766e-01 -3.1238e-02  2.8564e-01 -1.0514e-01
   9.0636e-01  3.1695e-01 -6.5648e-01  3.2521e-02  2.7298e-01 -1.5310e-01  4.5591e-01  1.8780e-01
   9.9554e-01 -1.4303e-01 -3.9783e-01  7.8327e-01  5.9864e-03  5.9943e-02  2.1847e-01  5.8644e-02
  -5.1259e-01  3.6053e-01 -3.2967e-01 -2.0133e-01 -1.0037e-02 -2.4647e-01  3.3688e-02 -9.0557e-01
  -3.3233e-02 -2.0612e-01 -2.2083e-01 -3.8905e-01  7.9650e-01  2.3235e-01  2.6399e-01  2.5961e-01]]
```

Notice that the three vectors above are all similar, because they represent similar sounds.

Run the above tools with the `--help`  option for further documentation. To compute many embeddings in bulk, see docs [here](doc/misc/batch_embed.md).

## Table of contents

1. [Set up environment and prepare data](doc/setup.md)
1. Train an acoustic frontend
   1. [Train and test HMM acoustic model](doc/fe_am/hmm.md)
   1. [Train DNN-HMM acoustic model](doc/fe_am/train_cd_dnn.md)
   1. [Train monophone DNN-HMM acoustic model](doc/fe_am/train_mono_dnn.md)
   1. [Test DNN-HMM acoustic models](doc/fe_am/test_dnn.md)
1. Train embedders for acoustic neighbor embeddings
   1. [Prepare data](doc/embed/dataprep.md)
   1. [Train embedders](doc/embed/train.md)
1. Experiments
   1. [Word classification](doc/exp/wordclassify.md)
   1. [OOV recovery](doc/exp/oovrecovery.md)
   1. [Dialect clustering](doc/exp/dialect.md)
   1. [Wake-up word confusion](doc/exp/wakeword.md) 
1. Miscellaneous
   1. [Standalone forced-alignment tool](doc/misc/forcealign.md)
   2. [Batch computation of embeddings](doc/misc/batch_embed.md)
1. [List of all downloadables](doc/downloads.md)

## BibTeX reference

If you find this work useful, please consider citing it as follows:
```
@misc{acn-embed,
      title={A Theoretical Framework for Acoustic Neighbor Embeddings}, 
      author={Woojay Jeon},
      year={2024},
      eprint={2412.02164},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2412.02164}, 
}
```
