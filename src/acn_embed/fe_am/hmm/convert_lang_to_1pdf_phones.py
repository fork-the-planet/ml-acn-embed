#!/usr/bin/env python3
"""
Takes a kaldi "lang_nosp" dir and converts it _inplace_ to a single-PDF-per-phone model,
suitable for a DNN model that has 1 output per (position-dependent) phone
"""

import argparse
import os
import pathlib
import sys

from acn_embed.util.phone_table import PhoneTable
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = pathlib.Path(__file__).parent.absolute()


def write_topo(fn, last_phone_id):
    """
    <Topology>
    <TopologyEntry>
    <ForPhones>
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 [...] 205
    </ForPhones>
    <State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>
    <State> 1 </State>
    </TopologyEntry>
    </Topology>
    """
    with open(fn, "w", encoding="utf8") as fobj:
        fobj.write("<Topology>\n<TopologyEntry>\n<ForPhones>\n")
        fobj.write(" ".join([str(x) for x in range(1, last_phone_id + 1)]) + "\n")
        fobj.write(
            "</ForPhones>\n<State> 0 <PdfClass> 0 <Transition> "
            "0 0.75 <Transition> 1 0.25 </State>\n"
        )
        fobj.write("<State> 1 </State>\n</TopologyEntry>\n</Topology>\n")


def write_roots(lang_dir, last_phone_id, phone_table):
    with open(os.path.join(lang_dir, "phones", "roots.txt"), "w", encoding="utf8") as fobj:
        fobj.write(
            "not-shared not-split "
            + " ".join([phone_table.id_to_pd_phone[x] for x in range(1, last_phone_id + 1)])
            + "\n"
        )
    with open(os.path.join(lang_dir, "phones", "roots.int"), "w", encoding="utf8") as fobj:
        fobj.write(
            "not-shared not-split "
            + " ".join([str(x) for x in range(1, last_phone_id + 1)])
            + "\n"
        )


def write_sets(lang_dir, last_phone_id, phone_table):
    with open(os.path.join(lang_dir, "phones", "sets.txt"), "w", encoding="utf8") as fobj:
        fobj.writelines(
            [(phone_table.id_to_pd_phone[x] + "\n") for x in range(1, last_phone_id + 1)]
        )
    with open(os.path.join(lang_dir, "phones", "sets.int"), "w", encoding="utf8") as fobj:
        fobj.writelines([(str(x) + "\n") for x in range(1, last_phone_id + 1)])


def main():
    parser = argparse.ArgumentParser(
        description="Converts a lang_nosp dir _inplace_ to a single-PDF-per-phone model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("lang_dir", action="store")
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    phone_table_fn = os.path.join(args.lang_dir, "phones.txt")
    phone_table = PhoneTable(filename=phone_table_fn)

    phone_ids = sorted(phone_table.id_to_pd_phone.keys())

    assert phone_table.id_to_pd_phone[0] == "<eps>"

    for phone_id in range(1, max(phone_ids) + 1):
        pd_phone = phone_table.id_to_pd_phone[phone_id]
        if pd_phone.startswith("#"):
            break
    last_phone_id = phone_id - 1
    assert last_phone_id > 0

    write_topo(os.path.join(args.lang_dir, "topo"), last_phone_id)
    write_roots(args.lang_dir, last_phone_id, phone_table)
    write_sets(args.lang_dir, last_phone_id, phone_table)


if __name__ == "__main__":
    main()
