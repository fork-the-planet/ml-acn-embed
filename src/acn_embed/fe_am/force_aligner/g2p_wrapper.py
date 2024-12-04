import subprocess
import tempfile
from pathlib import Path

from acn_embed.util.lexicon import Lexicon, fix_lexicon
from acn_embed.util.phone_table import PhoneTable
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class G2PWrapper:
    def __init__(self, lexicon: Path, g2p_model: Path, phone_table: Path):
        self.lexicon = Lexicon(path=lexicon)
        self.g2p_model = g2p_model
        self.phone_set = PhoneTable(filename=phone_table).get_pi_phone_set()

    def get_lexicon(self, words) -> Lexicon:
        """
        Given a list/set of words, returns a Lexicon object, either by looking up words in the
        internal lexicon or running G2P for words that are OOV.
        """
        word2prons = {}
        oovs = set()
        for word in words:
            if word in self.lexicon.word2prons:
                word2prons[word] = self.lexicon.word2prons[word].copy()
            else:
                oovs.add(word)
        lexicon = Lexicon(word2prons=word2prons)
        if oovs:
            oov_lexicon = self.get_lexicon_from_g2p(words)
            lexicon.merge_in(other_lexicon=oov_lexicon)

        # Fix any invalid prons (occur very rarely)
        fix_lexicon(lexicon, self.phone_set)

        return lexicon

    def get_lexicon_from_g2p(self, words) -> Lexicon:
        """
        Runs G2P to get a Lexicon for a list/set of words
        """
        _, tmpfn = tempfile.mkstemp()
        with open(tmpfn, "w", encoding="utf8") as fobj:
            fobj.writelines([word.upper() + "\n" for word in words])
        _, tmp_lex_fn = tempfile.mkstemp()
        LOGGER.debug(f"{tmp_lex_fn=}")
        with open(tmp_lex_fn, "w", encoding="utf8") as fobj:
            proc = subprocess.run(
                f"g2p.py --model {self.g2p_model} --apply {tmpfn} -V 0.8 --variants-number 3",
                shell=True,
                stdout=fobj,
                check=True,
            )
        assert proc.returncode == 0
        return Lexicon(tmp_lex_fn)
