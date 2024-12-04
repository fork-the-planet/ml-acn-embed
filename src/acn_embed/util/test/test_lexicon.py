#!/usr/bin/env python3

import filecmp
import os
import tempfile
import unittest
from pathlib import Path

from acn_embed.util.lexicon import Lexicon, get_merged_lexicon, fix_pron, fix_lexicon


class LexiconTest1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, cls.lexfile1 = tempfile.mkstemp()
        _, cls.lexfile2 = tempfile.mkstemp()

        # fmt: off
        with open(cls.lexfile1, "w", encoding="utf8") as fobj:
            fobj.write(
                "WORD1\tA B C\n"
                "WORD1\tD E F\n"
                "WORD2\tG H\n"
            )
        with open(cls.lexfile2, "w", encoding="utf8") as fobj:
            fobj.write(
                "WORD2\tI J K\n"
                "WORD3\tL M N O P\n"
                "WORD3(2)\tQ R S #another comment\n"
            )
        # fmt: on

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.lexfile1)
        os.remove(cls.lexfile2)

    def setUp(self):
        self.lexicon1 = Lexicon(Path(self.lexfile1))
        self.lexicon2 = Lexicon(Path(self.lexfile2))

    def test_word2prons(self):
        self.assertIn("WORD3", self.lexicon2.word2prons)
        self.assertNotIn("WORD3(2)", self.lexicon2.word2prons)
        self.assertCountEqual(self.lexicon2.word2prons["WORD3"], {"L M N O P", "Q R S"})

    def test_check_add_pron(self):
        self.lexicon1.add_pron("WORD3", "I J")
        self.assertIn("I J", self.lexicon1.word2prons["WORD3"])

    def test_merge_in(self):
        self.lexicon1.merge_in(self.lexicon2)
        self.assertCountEqual(self.lexicon1.word2prons.keys(), {"WORD1", "WORD2", "WORD3"})
        self.assertCountEqual(self.lexicon1.word2prons["WORD2"], {"G H"})

    def test_write(self):
        _, tmp_lexfile = tempfile.mkstemp()
        self.lexicon1.write(tmp_lexfile)
        self.assertTrue(filecmp.cmp(self.lexfile1, tmp_lexfile))

    def test_get_pron_set(self):
        self.assertCountEqual(self.lexicon1.get_pron_set(), {"A B C", "D E F", "G H"})

    def test_word_set(self):
        self.assertCountEqual(self.lexicon1.get_word_set(), {"WORD1", "WORD2"})

    def test_get_merged_lexicon(self):
        merged_lexicon = get_merged_lexicon([Path(self.lexfile1), Path(self.lexfile2)])
        self.assertCountEqual(merged_lexicon.word2prons.keys(), {"WORD1", "WORD2", "WORD3"})
        self.assertCountEqual(merged_lexicon.word2prons["WORD2"], {"G H"})

    def test_fix_pron(self):
        self.assertEqual(fix_pron("A B C", {"A", "B", "C"}), "A B C")
        self.assertEqual(fix_pron("A B C", {"A", "B0", "C"}), "A B0 C")
        self.assertIsNone(fix_pron("A B C", {"A", "B1", "C"}))

    def test_fix_lexicon(self):
        fixed = fix_lexicon(self.lexicon1, {"A", "B", "C", "D0", "E", "F", "G0", "H"})
        self.assertIn("A B C", fixed.word2prons["WORD1"])
        self.assertIn("D0 E F", fixed.word2prons["WORD1"])
        self.assertIn("G0 H", fixed.word2prons["WORD2"])

        # Invalid prons that couldn't be fixed should be dropped
        fixed = fix_lexicon(self.lexicon1, {"A0", "B", "C", "D1", "E", "F", "G1", "H"})
        self.assertIn("A0 B C", fixed.word2prons["WORD1"])
        self.assertEqual(len(fixed.word2prons["WORD1"]), 1)
        self.assertEqual(len(fixed.word2prons), 1)


if __name__ == "__main__":
    unittest.main()
