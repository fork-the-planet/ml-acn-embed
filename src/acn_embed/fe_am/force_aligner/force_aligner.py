#!/usr/bin/env python3
"""
"End-user" force alignment class
"""
import os
import re
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import kaldi_io
import torch

import acn_embed.fe_am.hmm.write_dict_nosp
from acn_embed.embed.data.phone_ali_to_tran import read_utt2_phone_len, get_word_frames
from acn_embed.fe_am.force_aligner.g2p_wrapper import G2PWrapper
from acn_embed.fe_am.nnet.infer.am_inference import AmInference, AmPriorType
from acn_embed.util.logger import get_logger
from acn_embed.util.phone_table import PhoneTable, phone_pd_to_pi
from acn_embed.util.word_table import WordTable

LOGGER = get_logger(__name__)
THIS_DIR = Path(__file__).parent


# pylint: disable=too-many-instance-attributes


class ForceAligner:
    def __init__(
        self,
        *,
        model: Path,
        lexicon: Path,
        g2p_model: Path,
        kaldi_bin_path: Path,
        kaldi_src_path: Path,
        frame_rate_ms=10,
        beam=10,
        retry_beam=40,
        device=None,
    ):
        """
        Initializes ForceAligner.

        model:
            Directory containing model files (am.pt, prior.pt, trans.mdl, etc.)

        lexicon:
            Pronunciation lexicon text file

        g2p_model:
            G2P model file

        beam, retry_beam:
            Do not change this unless you know what you're doing. A high beam will
            reduce force alignment failures, but on the other hand, if you are only
            able to get results with a very high beam, the resulting word timestamps
            are probably wrong and are not much use anyway.

        device:
            PyTorch device to use for AM inference. None to let PyTorch auto-detect.
            (e.g. torch.device("cpu"), torch.device("cuda"), etc.)
        """

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        if g2p_model is None:
            LOGGER.warning(
                "You have not provided a G2P model. Force alignment will fail if there are any "
                "words that don't have a pronunciation in the lexicon file."
            )

        # We're gonna chdir() later so we have to be careful to keep paths absolute
        self.g2p_wrapper = G2PWrapper(
            lexicon=lexicon.absolute(),
            g2p_model=g2p_model.absolute() if g2p_model else None,
            phone_table=(model / "lang_nosp" / "phones.txt"),
        )
        self.model_path = model.absolute()
        self.phone_table = PhoneTable(model / "lang_nosp" / "phones.txt")
        self.kaldi_bin_path = kaldi_bin_path.absolute()
        self.kaldi_src_path = kaldi_src_path.absolute()
        self.frame_rate_ms = frame_rate_ms
        self.beam = beam
        self.retry_beam = retry_beam
        self.device = device

        self.am_inference = AmInference(
            model_dir=model, boost_sil_log_prior=0, prior_type=AmPriorType.MARGIN, device=device
        )

        self.ms_per_frame = self.am_inference.ms_per_frame()

        # Create a temporary working dir
        self.work_dir = tempfile.mkdtemp()
        LOGGER.info(f"{self.work_dir=}")
        os.symlink(
            os.path.join(self.kaldi_src_path, "egs/wsj/s5/utils"),
            os.path.join(self.work_dir, "utils"),
        )
        Path(os.path.join(self.work_dir, "path.sh")).touch()  # make kaldi happy

        _, self.tmp_lexicon_fn = tempfile.mkstemp()
        _, self.tmp_am_output_file = tempfile.mkstemp()
        _, self.tmp_wordid_file = tempfile.mkstemp()
        self.tmp_dict_nosp_dir = tempfile.mkdtemp()
        self.tmp_lang_tmp_nosp_dir = tempfile.mkdtemp()
        self.tmp_lang_nosp_dir = tempfile.mkdtemp()

    def write_lang_nosp_dir(self, words):

        shutil.rmtree(self.tmp_dict_nosp_dir, ignore_errors=True)
        shutil.rmtree(self.tmp_lang_tmp_nosp_dir, ignore_errors=True)
        shutil.rmtree(self.tmp_lang_nosp_dir, ignore_errors=True)

        os.makedirs(self.tmp_dict_nosp_dir)
        os.makedirs(self.tmp_lang_tmp_nosp_dir)
        os.makedirs(self.tmp_lang_nosp_dir)

        self._sanity_check_words(words)

        lexicon = self.g2p_wrapper.get_lexicon(words)
        lexicon.write(self.tmp_lexicon_fn)

        acn_embed.fe_am.hmm.write_dict_nosp.write_dict_nosp(
            lexicon=Path(self.tmp_lexicon_fn),
            non_sil_phones=(THIS_DIR.parent.parent / "setup" / "non-sil-phones.json"),
            output_dir=Path(self.tmp_dict_nosp_dir),
        )

        proc = subprocess.run(
            f"{self.kaldi_src_path}/egs/wsj/s5/utils/prepare_lang.sh "
            f"{self.tmp_dict_nosp_dir} "
            f'"<UNK>" '
            f"{self.tmp_lang_tmp_nosp_dir} "
            f"{self.tmp_lang_nosp_dir} ",
            capture_output=True,
            shell=True,
            encoding="utf8",
            check=False,
        )

        if proc.returncode != 0:
            LOGGER.error(proc.stdout)
        return self.tmp_lang_nosp_dir

    # pylint: disable=too-many-locals
    def force_align(self, words: list, wav_fn=None, wav_bytes=None, fbank_tensor=None):
        """
        Does force-alignment using either a wav file, wav bytes, or an fbank tensor.

        words:
            A list of words

        wav_fn:
            A .wav file containing the audio. If not given, wav_bytes must be
            provided

        wav_bytes:
            Raw bytes of a .wav file

        fbank_tensor:
            Fbank feat tensor.

        Returns:
            A list of n-best results.
        """

        if wav_fn:
            wav_fn = wav_fn.absolute()

        os.chdir(self.work_dir)

        # pylint: disable=consider-using-generator
        assert (
            sum([(x is not None) for x in [wav_fn, wav_bytes, fbank_tensor]]) == 1
        ), "You must specify one (and only one) of wav_fn or wav_bytes or fbank_tensor"

        if wav_fn:
            output_nparr = self.am_inference.infer_from_wav_fn(wav_fn=wav_fn)
        elif wav_bytes:
            output_nparr = self.am_inference.infer_from_wav_bytes(wav_bytes=wav_bytes)
        elif fbank_tensor is not None:
            output_nparr = self.am_inference.infer_from_fbank_tensor(
                fbank_tensor=fbank_tensor, return_as_nparr=True
            )
        else:
            raise RuntimeError("You must specify either wav_fn, wav_bytes, or fbank_tensor")

        with open(self.tmp_am_output_file, "wb") as fobj:
            kaldi_io.write_mat(fobj, output_nparr, key="utt")

        lang_nosp_dir = self.write_lang_nosp_dir(words)

        word_table = WordTable(os.path.join(lang_nosp_dir, "words.txt"))

        wordids = " ".join([str(word_table.word_to_id[word]) for word in words])
        with open(self.tmp_wordid_file, "w", encoding="utf8") as fobj:
            fobj.write(f"utt {wordids}\n")

        proc = subprocess.run(
            args=(
                f"{self.kaldi_bin_path}/compile-train-graphs "
                f"--read-disambig-syms={self.model_path}/lang_nosp/phones/disambig.int "
                f"{self.model_path}/tree "
                f"{self.model_path}/trans.mdl "
                f'{lang_nosp_dir}/L.fst "ark,t:{self.tmp_wordid_file}" ark:- |'
                f"{self.kaldi_bin_path}/align-compiled-mapped "
                "--transition-scale=1.0 "
                "--acoustic-scale=0.1 "
                "--self-loop-scale=0.1 "
                f"--beam={self.beam} "
                f"--retry-beam={self.retry_beam} "
                "--careful=true "
                f"{self.model_path}/trans.mdl ark:- ark:{self.tmp_am_output_file} ark:- | "
                f"{self.kaldi_bin_path}/ali-to-phones --write-lengths {self.model_path}/trans.mdl "
                f"ark:- ark,t:- "
            ),
            shell=True,
            capture_output=True,
            encoding="utf8",
            check=False,
        )

        if not re.search(r"ali-to-phones.cc:\d+?\) Done 1 utterances", proc.stderr.strip()):
            LOGGER.error(proc.stderr)
            return None

        if proc.returncode != 0:
            return None

        utt2pl = read_utt2_phone_len(StringIO(proc.stdout))
        word_frames = get_word_frames(utt2pl["utt"], self.phone_table)

        tokens = [
            {
                "orth": words[idx],
                "pron": " ".join([phone_pd_to_pi(phone) for phone in word_frames[idx][2]]),
                "start_ms": word_frames[idx][0] * self.frame_rate_ms,
                "end_ms": word_frames[idx][1] * self.frame_rate_ms,
            }
            for idx in range(len(words))
        ]
        return tokens

    def _sanity_check_words(self, words):
        assert isinstance(words, list)
        assert all(words)
