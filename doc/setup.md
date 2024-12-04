# Initial setup

This document describes the environmental setup necessary for fully replicating all model training and experiments discussed in the paper.

This documentation has been tested using:
- zshell 5.8.1
- Python 3.10.14
- pytorch 2.3 & 2.5.1
- torchaudio 2.3 & 2.5.1

Build [Kaldi](https://github.com/kaldi-asr/kaldi) on your system. This documentation has been tested on Kaldi commit SHA `51744d32acc9b19457f9aaed09c83f218f4b5de4`.

You may need to install these dependencies:
```commandline
sudo apt install parallel ffmpeg swig
```

And you almost certainly want to work in a `venv` or `miniconda` environment.

Clone this repository, and from the root directory (where you see the [pyproject.toml](../pyproject.toml) file), install other Python dependencies (depending on your environment, you may need to do more):

```commandline
pip install -e ".[full]"
```

Appropriately edit `path.sh` and source it to set up your environment:

```commandline
source path.sh
```

Create dirs:

```commandline
mkdir -p $DATA $DNN $EXP $HMM $LOG $MISC $MODEL $RESOURCE $OUTPUT ${CKPT_DIR}
```

Get all the [downloadables](downloads.md) and put them in `$WORK/download`. The only "required" file is `short-utts.tgz`, but other files may be useful if you want to skip some steps or replicate our experiments exactly.

Extract WAV files from LibriHeavy:

```commandline
rm -f jobs
for DTYPE in test_clean test_clean_large test_other test_other_large large; do
    for SPLIT in `seq 1 8`; do
        echo "$ACN/setup/flac_to_utt_wav.py --libriheavy $LIBRIHEAVY --splits 8 --split-num ${SPLIT} --data-type $DTYPE" >> jobs
    done
done
parallel -j 8 < jobs
```

Download LibriSpeech resource files:

```commandline
mkdir -p $RESOURCE/
wget https://www.openslr.org/resources/11/librispeech-lexicon.txt -P $WORK/resource/
wget https://www.openslr.org/resources/11/g2p-model-5 -P $WORK/resource/
wget https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz -P $WORK/resource/
wget https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz -P $WORK/resource/
wget https://raw.githubusercontent.com/cmusphinx/cmudict/refs/heads/master/cmudict.dict -P $WORK/resource/
```

Convert cuts files to transcription file
```commandline
for DTYPE in large test_clean test_other; do
    $ACN/setup/cuts_to_tran.py \
        --src $LIBRIHEAVY/libriheavy_cuts_${DTYPE}.jsonl.gz \
        --dst $DATA/${DTYPE}.jsonl.gz
done
```

Generate transcription files for our custom datasets. Note that `tr7` is a 2,500-hour subset of `tra`:
```commandline
for DTYPE in tr7 tra trb cv exp; do
    $ACN/setup/cuts_to_tran.py \
        --shuffle \
        --src $LIBRIHEAVY/libriheavy_cuts_large.jsonl.gz \
        --list $WORK/download/short-utts/$DTYPE.txt \
        --dst $DATA/$DTYPE.jsonl.gz
done
```

The resulting training, development, and experiment data sets are:

| Data type | Number of utterances | Total hours |
|-----------|----------------------|-------------|
| TR-7      | 657,582              | 2,500       |
| TR-A      | 2,632,239            | 10,000      |
| TR-B      | 262,944              | 1,000       |
| CV        | 263,122              | 1,000       |
| EXP       | 263,431              | 1,000       |

Get a list of all the words in the training, dev, and experimental data. This will be used for HMM training and force-alignment of experimental data:

```commandline
$ACN/setup/get_words_from_tran.py \
    --src $DATA/tra.jsonl.gz \
          $DATA/trb.jsonl.gz \
          $DATA/cv.jsonl.gz \
          $DATA/exp.jsonl.gz \
    --dst $MISC/forcealign-words.txt
```

Get lexicon for the above word list, running G2P when necessary. This can take a while.

```commandline
$ACN/setup/get_lexicon.py \
    --dst-lexicon $MISC/forcealign-lexicon.txt \
    --src-lexicon $RESOURCE/cmudict.dict \
                  $RESOURCE/librispeech-lexicon.txt \
    --g2p $RESOURCE/g2p-model-5 \
    --list $MISC/forcealign-words.txt \
    --phones $ACN/setup/non-sil-phones.json
```

This results in about 386,853 words, 720,249 prons, and 615,497 unique prons in `$MISC/forcealign-lexicon.txt`.

Get a list of all the words in the `3-gram.pruned.1e-7.arpa.gz` LM. This will be used for HMM testing:

```commandline
$ACN/setup/get_words_from_lm.py \
    --arpa $RESOURCE/3-gram.pruned.1e-7.arpa.gz \
    --dst $MISC/testing-words.txt
```

Get lexicon for the above word list.

```commandline
$ACN/setup/get_lexicon.py \
    --dst-lexicon $MISC/testing-lexicon.txt \
    --src-lexicon $RESOURCE/cmudict.dict \
                  $RESOURCE/librispeech-lexicon.txt \
                  $MISC/forcealign-lexicon.txt \
    --g2p $RESOURCE/g2p-model-5 \
    --list $MISC/testing-words.txt \
    --phones $ACN/setup/non-sil-phones.json
```

This results in about 200,000 words, 206,093 prons, and 176,921 unique prons in `$MISC/testing-lexicon.txt`.

Get a sanitized list of all words in the LibriHeavy dataset, for use in experiments:

```commandline
$ACN/setup/get_words_from_tran.py --sanitize \
    --src $DATA/large.jsonl.gz \
    --dst $MISC/all-words.txt
```

Get lexicon for the above word list, running G2P when necessary. This can take significant time:

```commandline
$ACN/setup/get_lexicon.py \
    --dst-lexicon $MISC/all-lexicon.txt \
    --src-lexicon $RESOURCE/cmudict.dict \
                  $RESOURCE/librispeech-lexicon.txt \
                  $MISC/forcealign-lexicon.txt \
                  $MISC/testing-lexicon.txt \
    --g2p $RESOURCE/g2p-model-5 \
    --list $MISC/all-words.txt \
    --phones $ACN/setup/non-sil-phones.json
```

This results in about 956,102 words, 2,185,471 prons, 1,806,389 unique prons in `$MISC/all-lexicon.txt`.
