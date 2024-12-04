#!/usr/bin/env python3
import re
import subprocess
import tempfile
from pathlib import Path

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class Lexicon:
    MULTPRON = re.compile(r"\(\d+\)$")

    def __init__(self, path: Path = None, word2prons: dict = None):
        if path is None and word2prons is None:
            self.word2prons = {}
            return
        if word2prons is not None:
            self.word2prons = word2prons
            return
        numeric = re.compile(r"[\d\.]+")
        self.word2prons = {}
        with open(path, "r", encoding="utf8") as fobj:
            for line in fobj:
                toks = line.strip().split()
                if not toks:
                    continue
                word = toks[0]

                word = self.MULTPRON.sub("", word.upper())

                if len(toks) >= 3 and numeric.match(toks[1]) and numeric.match(toks[2]):
                    pron = " ".join(toks[3:])
                else:
                    pron = " ".join(toks[1:])

                if "#" in pron:
                    pron = pron[: pron.index("#")].strip()

                if not pron:
                    continue
                if word not in self.word2prons:
                    self.word2prons[word] = set()
                self.word2prons[word].add(pron)

    def add_pron(self, word, pron):
        if word not in self.word2prons:
            self.word2prons[word] = set()
        self.word2prons[word].add(pron)

    def merge_in(self, other_lexicon):
        for word, prons in other_lexicon.word2prons.items():
            if word not in self.word2prons:
                self.word2prons[word] = prons

    def write(self, filename):
        tot_prons = 0
        unique_prons = set()
        with open(filename, "w", encoding="utf8") as fobj:
            for word in sorted(self.word2prons.keys()):
                for pron in sorted(self.word2prons[word]):
                    fobj.write(f"{word}\t{pron}\n")
                    unique_prons.add(pron)
                    tot_prons += 1
        LOGGER.info(
            f"Wrote {len(self.word2prons):,d} words, "
            f"{tot_prons:,d} prons, "
            f"{len(unique_prons):,d} unique prons "
            f"to {filename}"
        )

    def get_pron_set(self):
        prons = set()
        for _prons in self.word2prons.values():
            prons |= _prons
        return prons

    def get_word_set(self):
        return set(self.word2prons.keys())


def fix_pron(pron, pi_phone_set: set):
    """
    Try to correct invalid prons outputted by the G2P on rare occasions
    """
    phones = []
    for phone in pron.split():
        if phone in pi_phone_set:
            phones.append(phone)
            continue
        phone = phone + "0"
        if phone in pi_phone_set:
            phones.append(phone)
            continue
        LOGGER.info(f"Failed to fix invalid pron: {pron}")
        return None
    return " ".join(phones)


def fix_lexicon(lex: Lexicon, phoneset: set):
    fixed_lex = Lexicon(word2prons={})
    for word, prons in lex.word2prons.items():
        for pron in prons:
            fixed_pron = fix_pron(pron, phoneset)
            if fixed_pron is not None:
                if fixed_pron != pron:
                    LOGGER.info(f"Fixed {pron} -> {fixed_pron}")
                fixed_lex.add_pron(word, fixed_pron)
    return fixed_lex


def get_merged_lexicon(paths: list) -> Lexicon:
    lex = Lexicon()
    for path in paths:
        lex.merge_in(Lexicon(path=path))
    return lex


def get_lexicon_from_g2p(words, g2p):
    LOGGER.info(f"Running g2p on {len(words)} words")
    _, tmpfn = tempfile.mkstemp()
    with open(tmpfn, "w", encoding="utf8") as fobj:
        fobj.writelines([word.upper() + "\n" for word in words])
    _, tmp_lex_fn = tempfile.mkstemp()
    LOGGER.info(f"{tmp_lex_fn=}")
    with open(tmp_lex_fn, "w", encoding="utf8") as fobj:
        proc = subprocess.run(
            f"g2p.py --model {g2p} --apply {tmpfn} -V 0.8 --variants-number 3",
            shell=True,
            stdout=fobj,
            check=True,
        )
    assert proc.returncode == 0
    return Lexicon(path=Path(tmp_lex_fn))
