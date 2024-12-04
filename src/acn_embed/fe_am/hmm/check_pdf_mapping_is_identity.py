#!/usr/bin/env python3
import argparse
import sys

from acn_embed.fe_am.hmm.get_sil_pdfs import get_phone_id_to_pdf_ids
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=("Check that the mapping from PDF to Phone ID is an identity transformation"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("trans_model", action="store", type=str)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    phone_id_to_pdf_ids, max_phone_id, _ = get_phone_id_to_pdf_ids(args.trans_model)

    assert len(phone_id_to_pdf_ids) == max_phone_id
    for phone_id in range(1, max_phone_id + 1):
        assert phone_id in phone_id_to_pdf_ids
        assert len(phone_id_to_pdf_ids[phone_id]) == 1
        assert list(phone_id_to_pdf_ids[phone_id])[0] == phone_id - 1

    print("Mapping is identity.")


if __name__ == "__main__":
    main()
