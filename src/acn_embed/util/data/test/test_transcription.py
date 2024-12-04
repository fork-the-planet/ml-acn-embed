#!/usr/bin/env python3
import filecmp
import gzip
import os
import tempfile
import unittest
from ast import literal_eval
from pathlib import Path

from acn_embed.util.data.transcription import TranscriptionWriter, read_tran_utts, write_tran_utts


class TranscriptionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, cls.tranfile = tempfile.mkstemp(suffix=".jsonl")
        _, cls.tranfile_gz = tempfile.mkstemp(suffix=".jsonl.gz")

        cls.strings = [
            '{"type": "utt", "utt_id": "abcd", "nbest": ["hello", "hi", "how are you"]}\n',
            '{"type": "utt", "utt_id": "efgh", "nbest": ["I\'m", "fine"]}\n',
            '{"type": "utt", "utt_id": "ijkl", "nbest": ["thank", "you"]}\n',
        ]
        cls.content = [literal_eval(string) for string in cls.strings]

        with open(cls.tranfile, "w", encoding="utf8") as fobj:
            fobj.writelines(cls.strings)
        with gzip.open(cls.tranfile_gz, "wt", encoding="utf8") as fobj:
            fobj.writelines(cls.strings)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.tranfile)
        os.remove(cls.tranfile_gz)

    def test_read_utts_gz(self):
        utts = []
        for utt in read_tran_utts(self.tranfile_gz):
            utts.append(utt)
        self.assertEqual(utts[0]["utt_id"], "abcd")
        self.assertEqual(utts[1]["utt_id"], "efgh")
        self.assertEqual(utts[2]["utt_id"], "ijkl")

    def test_write_utts(self):
        _, tmp_tranfile = tempfile.mkstemp(suffix=".jsonl")
        write_tran_utts(self.content, Path(tmp_tranfile))
        self.assertTrue(filecmp.cmp(self.tranfile, tmp_tranfile))

    def test_write_utts_gz(self):
        _, tmp_tranfile_gz = tempfile.mkstemp(suffix=".jsonl.gz")
        write_tran_utts(self.content, Path(tmp_tranfile_gz))
        with gzip.open(tmp_tranfile_gz, "rt") as fobj1:
            with gzip.open(self.tranfile_gz, "rt") as fobj2:
                self.assertEqual(fobj1.read(), fobj2.read())

    def test_writer(self):
        _, tmp_tranfile = tempfile.mkstemp(suffix=".jsonl")
        with TranscriptionWriter(tmp_tranfile) as writer:
            for num, utt in enumerate(read_tran_utts(self.tranfile_gz)):
                writer.write(utt)
                self.assertEqual(num + 1, writer.num_written)
        self.assertTrue(filecmp.cmp(self.tranfile, tmp_tranfile))


if __name__ == "__main__":
    unittest.main()
