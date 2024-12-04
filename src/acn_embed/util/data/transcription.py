#!/usr/bin/env python3

import gzip
import json
from pathlib import Path

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def read_tran_utts(path: Path):
    assert str(path).endswith(".jsonl.gz")
    fobj = gzip.open(path, "rt", encoding="utf8")
    for line in fobj:
        line = line.strip()
        if line:
            dic = json.loads(line.strip())
            yield dic
    fobj.close()


def write_tran_utts(utts: list, path: Path):
    with TranscriptionWriter(path) as writer:
        for utt in utts:
            writer.write(utt)


class TranscriptionWriter:
    def __init__(self, path: Path):
        self.path = path
        assert str(path).endswith(".jsonl") or str(path).endswith(
            ".jsonl.gz"
        ), "Filename must end with .jsonl or .jsonl.gz"
        self._fobj = None
        self._nutts = 0

    def __enter__(self):
        LOGGER.info(f"Streaming to {self.path}")
        if str(self.path).endswith(".jsonl"):
            self._fobj = open(self.path, "w", encoding="utf8")
        else:
            self._fobj = gzip.open(self.path, "wt", encoding="utf8")
        return self

    def write(self, utt):
        json.dump(utt, self._fobj, indent=None)
        self._fobj.write("\n")
        self._nutts += 1

    @property
    def num_written(self):
        return self._nutts

    def __exit__(self, *exc):
        self._fobj.close()
        LOGGER.info(f"Wrote {self._nutts} utts to {self.path}")
