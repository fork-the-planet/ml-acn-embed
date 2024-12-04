#!/usr/bin/env python3
import argparse
import sys

import kaldi_io
import numpy as np

from acn_embed.util.phone_table import PhoneTable, phone_pd_to_pi
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert phone IDs from PD to 0-based PI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pd-table", action="store", type=str)
    parser.add_argument("--pi-table", action="store", type=str)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    pd_table = PhoneTable(args.pd_table)
    pi_table = PhoneTable(args.pi_table)

    pd_ids = sorted(pd_table.id_to_pd_phone.keys())
    pd_id_to_pi_id = {}
    for pd_id in pd_ids:
        if pd_id == 0:
            continue
        pd_phone = pd_table.id_to_pd_phone[pd_id]
        if pd_phone.startswith("#"):
            continue
        pi_id = pi_table.pd_phone_to_id[phone_pd_to_pi(pd_phone)]
        pd_id_to_pi_id[pd_id] = pi_id

    for key, nparr in kaldi_io.read_vec_int_ark(sys.stdin.buffer):
        kaldi_io.write_vec_int(
            sys.stdout.buffer,
            np.array([pd_id_to_pi_id[_pd] - 1 for _pd in nparr]).astype(np.int32),
            key,
        )


if __name__ == "__main__":
    main()
