#!/usr/bin/env bash

nj=4
cmd=run.pl

echo "$0 $@"

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: steps/ali_to_pdf.sh <model-dir> <align-dir>"
   exit 1;
fi

modeldir=$1
aligndir=$2

for f in $modeldir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

$cmd JOB=1:$nj $aligndir/log/ali_to_pdf.JOB.log \
  ali-to-pdf $modeldir/final.mdl "ark:gunzip -c $aligndir/ali.JOB.gz|" "ark,t:|gzip -c >$aligndir/pdf.JOB.gz" || exit 1;
echo "$0: done aligning data."
