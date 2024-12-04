#!/usr/bin/env python3

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_phone_id_to_pdf_ids(trans_model):
    """
    Both phone IDs and PDF IDs start at 0, but note that phone ID 0 is <eps>
    """
    phone_id_to_pdf_ids = defaultdict(set)
    max_phone_id = -1
    max_pdf_id = -1
    with open(trans_model, "r", encoding="utf8") as fobj:
        state = 0
        for line in fobj:
            if state == 0:
                if line.startswith("<Triples>"):
                    state = 1
                continue
            if state == 1:
                if line.startswith("</Triples>"):
                    break
                phone_id, _, pdf_id = [int(x) for x in line.strip().split()]
                phone_id_to_pdf_ids[phone_id].add(pdf_id)
                max_pdf_id = max(max_pdf_id, pdf_id)
                max_phone_id = max(max_phone_id, phone_id)
    return dict(phone_id_to_pdf_ids), max_phone_id, max_pdf_id


def main():
    parser = argparse.ArgumentParser(
        description=("Get PDF IDs for silence phones from a transition model"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trans-model", action="store", type=Path, required=True)
    parser.add_argument("--sil-phones-int", action="store", type=Path, required=True)
    parser.add_argument("--output", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    with open(args.sil_phones_int, "r", encoding="utf8") as fobj:
        sil_phone_ids = [int(x.strip()) for x in fobj.readlines() if x.strip()]

    phone_id_to_pdf_ids, _, _ = get_phone_id_to_pdf_ids(args.trans_model)

    sil_pdf_ids = set()

    for phone_id in sil_phone_ids:
        sil_pdf_ids |= phone_id_to_pdf_ids[phone_id]

    with open(args.output, "w", encoding="utf8") as fobj:
        for pdf_id in sorted(sil_pdf_ids):
            fobj.write(f"{pdf_id}\n")


if __name__ == "__main__":
    main()
