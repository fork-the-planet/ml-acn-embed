import enum
import io
from pathlib import Path

import numpy as np
import torch
import torchaudio

import acn_embed.fe_am.nnet.model.get_model
from acn_embed.util.base_inference_input import BaseInferenceInput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class AmPriorType(enum.Enum):
    NONE = "NONE"
    MARGIN = "MARGIN"


class AmInference:
    def __init__(
        self,
        model_dir: Path,
        boost_sil_log_prior=None,
        prior_type: AmPriorType = AmPriorType.MARGIN,
        device=None,
    ):
        """
        Initializes acoustic model inference.

        model_dir:
            The model directory. Expects am.pt & prior.pt for AM inference.
            Expects sil_pdf_ids.txt to do silence boosting.

        boost_sil_log_prior:
            Extra log prior to add to the outputs of silence PDFs. Default is no silence boosting.

        prior_type:
            AmPriorType.MARGIN: Use log sum of outputs over the training data.
            AmPriorType.NONE: Don't do prior division (i.e., output posteriors)

        device:
            The device to do inference on (cpu or cuda). Specify None to auto-detect.
        """

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        LOGGER.info(f"device={self.device}")

        self.boost_sil_log_prior = boost_sil_log_prior

        with open(model_dir / "sil_pdf_ids.txt", "r", encoding="utf8") as fobj:
            self.sil_pdf_ids = [int(x.strip()) for x in fobj.readlines() if x.strip()]

        if self.boost_sil_log_prior:
            LOGGER.info(f"Boosting silence PDFs {self.sil_pdf_ids} by {self.boost_sil_log_prior}")

        fromfile = torch.load(model_dir / "am.pt", map_location=self.device, weights_only=True)
        self.model_spec = fromfile["model_spec"]
        self.model = acn_embed.fe_am.nnet.model.get_model.get(model_spec=self.model_spec).to(
            device=self.device
        )
        self.model.load_state_dict(fromfile["model_state_dict"])
        self.model.eval()

        self.log_prior = None  # Output posteriors by default

        assert isinstance(prior_type, AmPriorType)
        if prior_type == AmPriorType.MARGIN:
            fromfile = torch.load(
                model_dir / "prior.pt", map_location=self.device, weights_only=True
            )
            self.log_prior = fromfile["output_log_sum"] - torch.log(
                torch.tensor(fromfile["n"])
            ).to(device=self.device)

    def infer_from_wav_fn(self, wav_fn: Path, no_dither=False, return_as_nparr=True):
        waveform, sampfreq = torchaudio.load(wav_fn)
        return self.infer_from_wav_tensor(
            waveform, sampfreq, no_dither, return_as_nparr=return_as_nparr
        )

    def infer_from_wav_bytes(self, wav_bytes, no_dither=False, return_as_nparr=True):
        waveform, sampfreq = torchaudio.load(io.BytesIO(wav_bytes))
        return self.infer_from_wav_tensor(
            waveform, sampfreq, no_dither, return_as_nparr=return_as_nparr
        )

    def infer_from_wav_tensor(self, waveform, sampfreq, no_dither=False, return_as_nparr=True):

        if sampfreq != 16000:
            resampler = torchaudio.transforms.Resample(sampfreq, 16000, dtype=waveform.dtype)
            waveform = resampler(waveform)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=waveform,
            sample_frequency=16000,
            num_mel_bins=80,
            dither=0.0 if no_dither else 1.0 / 32767.0,  # Equivalent of 1.0 w/ 16-bit wav in Kaldi
        ).to(torch.float32)
        return self.infer_from_fbank_tensor(fbank, return_as_nparr=return_as_nparr)

    def infer_from_fbank_tensor(self, fbank_tensor, return_as_nparr=True):

        assert fbank_tensor.ndim == 2

        if fbank_tensor.shape[0] < self.model.min_input_len:
            LOGGER.info(
                f"Num rows = {fbank_tensor.shape[0]} is less than "
                f"min len {self.model.min_input_len}. "
                f"Inference failed."
            )
            return None

        with torch.no_grad():

            output_t, _ = self.model.forward(
                for_input=BaseInferenceInput(
                    input_t=fbank_tensor.unsqueeze(0),
                    input_len_t=torch.tensor([fbank_tensor.shape[0]]),
                ).to(device=self.device)
            )

            output_t = torch.log_softmax(output_t[0], dim=-1)

            if self.log_prior is not None:
                output_t = output_t - self.log_prior

            if self.boost_sil_log_prior:
                output_t[:, self.sil_pdf_ids] += self.boost_sil_log_prior

            if return_as_nparr:
                return output_t.detach().cpu().numpy().astype(np.float32)

            return output_t

    def infer_from_fbank_batch_tensor(self, fbank_tensor, input_len_tensor):

        assert fbank_tensor.ndim == 3

        if fbank_tensor.shape[1] < self.model.min_input_len:
            LOGGER.info(
                f"Num rows = {fbank_tensor.shape[1]} is less than "
                f"min len {self.model.min_input_len}. "
                f"Inference failed."
            )
            return None

        with torch.no_grad():

            output_t, output_len_t = self.model.forward(
                for_input=BaseInferenceInput(
                    input_t=fbank_tensor, input_len_t=input_len_tensor
                ).to(device=self.device)
            )

            output_t = torch.log_softmax(output_t, dim=-1)

            if self.log_prior is not None:
                output_t = output_t - self.log_prior

            if self.boost_sil_log_prior:
                output_t[:, :, self.sil_pdf_ids] += self.boost_sil_log_prior

            return output_t, output_len_t

    def ms_per_frame(self):
        return self.model.get_frame_ms()
