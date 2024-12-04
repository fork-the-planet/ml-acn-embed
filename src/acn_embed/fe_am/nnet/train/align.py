#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess
import sys

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = pathlib.Path(__file__).parent.absolute()


def main():
    parser = argparse.ArgumentParser(
        description="Get alignments using a DNN-HMM acoustic model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--boost-sil-log-prior", action="store", type=float, default=0)
    parser.add_argument("--ref-tran", type=str, default=None, required=True)
    parser.add_argument("--h5", type=str, default=None, required=True)
    parser.add_argument("--lang-dir", type=str, default=None, required=True)
    parser.add_argument("--model-dir", type=str, default=None, required=True)
    parser.add_argument("--beam", type=float, default=10.0)
    parser.add_argument("--retry-beam", type=float, default=40.0)
    parser.add_argument("--output-gz", type=str, required=True)
    parser.add_argument(
        "--output-type",
        choices=["tid", "pid", "pd_phone_len", "pi_phone0id"],
        default="tid",
        help="Output type. Transition IDs, PDF IDs, pos-dep phone ID & len, "
        "or pos-indep phone0 ID (start at 0 instead of 1)",
    )
    parser.add_argument(
        "--pd-table", type=str, default=None, required=False, help="Required for pi_phone0id"
    )
    parser.add_argument(
        "--pi-table", type=str, default=None, required=False, help="Required for pi_phone0id"
    )
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    subprocess.run(
        args=(
            f"{THIS_DIR}/../../data/get_text_from_tran.py "
            f"--tran {args.ref_tran} "
            f"--output {args.ref_tran}.text "
            f"--sort-utt-ids "
        ),
        shell=True,
        check=True,
    )

    feats = (
        f"ark,s,cs:{THIS_DIR}/../infer/infer_am.py "
        f"--sort-utt-ids "
        f"--h5 {args.h5} "
        f"--tran {args.ref_tran} "
        f"--model-dir {args.model_dir} "
        f"--boost-sil-log-prior {args.boost_sil_log_prior} |"
    )

    sym2int = os.path.join(os.environ["KALDI_SRC"], "egs/wsj/s5/utils/sym2int.pl")
    tra = (
        f"ark:{sym2int} "
        "--map-oov 1 "
        "-f 2- "
        f"{args.lang_dir}/words.txt "
        f"{args.ref_tran}.text |"
    )

    if args.output_type == "tid":
        output = f'"ark,t:|gzip -c >{args.output_gz}"'
    elif args.output_type == "pid":
        output = (
            f"ark:- | ali-to-pdf {args.model_dir}/trans.mdl "
            f'ark:- "ark,t:|gzip -c >{args.output_gz}"'
        )
    elif args.output_type == "pd_phone_len":
        output = (
            f"ark:- | ali-to-phones --write-lengths {args.model_dir}/trans.mdl "
            "ark:- ark,t:- | "
            f"gzip -c >{args.output_gz}"
        )
    elif args.output_type == "pi_phone0id":
        output = (
            f"ark:- | ali-to-phones --per-frame {args.model_dir}/trans.mdl ark:- ark,t:- | "
            f"{THIS_DIR}/../../hmm/remap_phone_ids_pd_to_0pi.py "
            f"--pd-table {args.pd_table} --pi-table {args.pi_table} | "
            f"gzip -c >{args.output_gz}"
        )
    else:
        raise RuntimeError("coding error")

    subprocess.run(
        args=(
            "compile-train-graphs "
            f"--read-disambig-syms={args.lang_dir}/phones/disambig.int "
            f"{args.model_dir}/tree "
            f"{args.model_dir}/trans.mdl "
            f'{args.lang_dir}/L.fst "{tra}" ark:- |'
            "align-compiled-mapped "
            "--transition-scale=1.0 "
            "--acoustic-scale=0.1 "
            "--self-loop-scale=0.1 "
            f"--beam={args.beam} "
            f"--retry-beam={args.retry_beam} "
            "--careful=true "
            f'{args.model_dir}/trans.mdl ark:- "{feats}" {output}'
        ),
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    main()
