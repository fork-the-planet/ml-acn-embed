####### Edit appropriately to fit your environment

export LIBRIHEAVY=/path/to/libriheavy
export TIMIT=/path/to/TIMIT
export OPENFST_BIN=/path/to/openfst/executables
export KALDI_BIN=/path/to/kaldi/executables
export KALDI_LIB=/path/to/kaldi/libraries   # Depending on how you built kaldi, may not need to set
export KALDI_SRC=/path/to/kaldi/src
export KALDI_ROOT=$KALDI_BIN
export ACN=/path/to/ml-acn-embed/src/acn_embed
export WORK=/my/work/path
# Choose from en.wikipedia.org/wiki/List_of_tz_database_time_zones
export LOG_TIME_ZONE="America/Los_Angeles"

# Directory for storing checkpoints during model training.
# Should be emptied before training a new model.
export CKPT_DIR=/mnt/ckpt

# Directory for storing models during training.
# Should be emptied before training a new model.
export OUTPUT=/mnt/output


######## Do not edit below this line unless you know what you're doing #######

alias csort="LC_ALL=C sort"
export LC_ALL=C
export DATA=$WORK/data
export DNN=$WORK/dnn
export EXP=$WORK/exp
export HMM=$WORK/hmm
export LOG=$WORK/log
export MISC=$WORK/misc
export MODEL=$WORK/model
export RESOURCE=$WORK/resource
export PYTHONPATH=$ACN/../:$PYTHONPATH
export PATH=$OPENFST_BIN:$KALDI_BIN:$HMM/utils:$PATH
export LD_LIBRARY_PATH=$KALDI_LIB:$LD_LIBRARY_PATH
export EXPWC=$EXP/wordclassify
