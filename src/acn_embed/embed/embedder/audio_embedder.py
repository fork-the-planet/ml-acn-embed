#!/usr/bin/env python3
import argparse
import bisect
from pathlib import Path

import numpy as np
import torch

import acn_embed.embed.model.get_model
from acn_embed.fe_am.nnet.infer.am_inference import AmInference, AmPriorType
from acn_embed.util.base_inference_input import BaseInferenceInput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class AudioEmbedder:
    def __init__(self, model_dir: Path, device=None):
        """
        Initialize an audio embedder for computing Acoustic Neighbor Embeddings

        model_dir:
            Directory containing "audio.pt" file.
            There should also exist a sibling directory "hybrid-mono"

        device:
            The device to run inference on. Will auto-detect if not specified.
        """
        assert model_dir.is_dir()

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

        self.model_dir = model_dir
        self.frontend_am_inference = AmInference(
            model_dir=model_dir.parent / "hybrid-mono",
            boost_sil_log_prior=0,  # We don't do silence boosting when running the audio embedder
            prior_type=AmPriorType.NONE,  # We only care about normalized posteriors
            device=self.device,
        )

        self.sil_score_idx = self.frontend_am_inference.sil_pdf_ids[0]

        embedder_fn = model_dir / "audio.pt"
        from_file = torch.load(embedder_fn, map_location=self.device, weights_only=True)
        self.model_spec = from_file["model_spec"]
        self.model = acn_embed.embed.model.get_model.get(model_spec=self.model_spec).to(
            self.device
        )
        self.model.load_state_dict(from_file["model_state_dict"])
        self.model.eval()

    def get_embedding(
        self,
        *,
        wav_fn: Path = None,
        wav_bytes=None,
        fbank_tensor: torch.Tensor = None,
        amfeat_tensor: torch.Tensor = None,
        segments: np.ndarray = None,
        trim_sil_thres=-0.1,
        no_dither: bool = False,
    ):
        """
        Get Acoustic Neighbor Embeddings for the given segment(s)

        wav_fn:
            A .wav file containing the audio. If not given, wav_bytes must be
            provided

        wav_bytes:
            Raw bytes of a .wav file

        fbank_tensor:
            Mel-filterbank feature tensor. See get_embedding_from_fbank_tensor()
            for how fbank tensor is computed from wav.

        amfeat_tensor:
            AM feat tensor, if you somehow managed to compute them yourself.

        segments:
            A 2-d numpy array with shape (num_segments, 2) where each row stores the (start, end)
            of each segment where start and end are in milliseconds. If None, will use entire wav
            file.

        trim_sil_thres:
            A float <= 0. This threshold is used to trim the front and back of each segment so that
            contiguous silence frames at the beginning and end are dropped. The lower the value,
            the more aggressive the trimming. The maximum value (0) results in no trimming.
            The default is -0.1.

        no_dither:
            Set to True to turn off dithering, which is adding tiny random noise to the audio to
            avoid numerical issues caused by zeros. The tiny noise will also make slight random
            changes to the embeddings, which is generally not an issue. You can turn off dithering
            if you know there are no running zeros in your audio files.

        Returns:
            A 2-d PyTorch tensor where each row stores an embedding for one segment.
            The number of rows should be the same as the number of segments.
            None if embedder failed for some reason.
        """
        with torch.no_grad():
            if wav_fn is not None:
                amfeat_tensor = self.frontend_am_inference.infer_from_wav_fn(
                    wav_fn, no_dither=no_dither, return_as_nparr=False
                )
            elif wav_bytes is not None:
                amfeat_tensor = self.frontend_am_inference.infer_from_wav_bytes(
                    wav_bytes, no_dither=no_dither, return_as_nparr=False
                )
            elif fbank_tensor is not None:
                amfeat_tensor = self.frontend_am_inference.infer_from_fbank_tensor(
                    fbank_tensor, return_as_nparr=False
                )
            else:
                assert amfeat_tensor is not None

            if segments is not None:
                assert segments.ndim == 2
                assert segments.shape[1] == 2
                num_segments = segments.shape[0]
                segments = self.ms_to_amfeat_frame(segments)
                segments = np.minimum(segments, amfeat_tensor.shape[0])
            else:
                # segments not specified, so use entire audio
                segments = np.array([[0, amfeat_tensor.shape[0]]], dtype=np.int32)
                num_segments = 1

            if trim_sil_thres < 0:
                segments = self.trim_segments(segments, amfeat_tensor, trim_sil_thres)
                if segments is None:
                    return None

            segment_len = segments[:, 1] - segments[:, 0]
            max_segment_len = np.max(segment_len)
            amfeats = torch.zeros(
                size=(num_segments, max_segment_len, amfeat_tensor.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            for batch, segment in enumerate(segments):
                amfeats[batch, : segment[1] - segment[0], :] = amfeat_tensor[
                    segment[0] : segment[1], :
                ]
            return self.model.forward(
                BaseInferenceInput(
                    input_t=amfeats,
                    input_len_t=torch.LongTensor(segment_len, device=torch.device("cpu")),
                )
            )

    def trim_segments(self, segments, amfeat_tensor, sil_thres):
        nonsil_idx = (
            torch.nonzero(amfeat_tensor[:, self.sil_score_idx] <= sil_thres).squeeze(1).tolist()
        )
        if not nonsil_idx:
            LOGGER.warning("Did not find any non-silence frames. trim_sil_thres too high?")
            return None
        trimmed_segments = []
        for segment in segments:
            if segment[0] > nonsil_idx[-1]:
                LOGGER.warning(
                    "Segment outside of non-silence range. "
                    "Bug in transcription or segmentation?"
                )
                return None
            seg0 = nonsil_idx[bisect.bisect_left(nonsil_idx, segment[0])]
            seg1 = nonsil_idx[bisect.bisect_right(nonsil_idx, segment[1] - 1) - 1] + 1
            if not (
                (segment[0] <= seg0 < segment[1])
                and (segment[0] < seg1 <= segment[1])
                and (seg0 < seg1)
            ):
                LOGGER.warning(
                    "Failed to get valid non-silence frames. "
                    "trim_sil_thres too high or segments too short?"
                )
                return None
            trimmed_segments.append([seg0, seg1])
        LOGGER.debug("Trimmed %s to %s", segments.tolist(), trimmed_segments)
        return np.array(trimmed_segments, dtype=np.int32)

    def ms_to_fbank_frame(self, ms):
        """
        Convert milliseconds to frame
        """
        return np.rint(ms / self.frontend_am_inference.ms_per_frame()).astype(np.int32)

    def ms_to_amfeat_frame(self, ms):
        """
        Convert milliseconds to amfeat frame
        """
        return self.ms_to_fbank_frame(ms)


def main():
    parser = argparse.ArgumentParser(
        description='Run audio ("f") embedder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Model directory containing embedder-audio/ and hybrid-full/",
    )
    parser.add_argument("--wav", type=Path, required=True, help=".wav filename")
    parser.add_argument(
        "--start-ms",
        type=int,
        default=None,
        help="Segment start time (ms). If not specified, the entire wav file will be used.",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=None,
        help="Segment end time (ms). If not specified, the entire wav file will be used.",
    )
    parser.add_argument(
        "--trim-sil-thres",
        type=float,
        default=-0.1,
        help=(
            "Threshold (must be <=0) used to trim the front and back of each segment so that "
            "contiguous silence frames at the beginning and end are dropped. The lower the value, "
            "the more aggressive the trimming. The maximum value (0) results in no trimming. "
            "The default is -0.1."
        ),
    )
    args = parser.parse_args()

    if args.start_ms is None or args.end_ms is None:
        segments = None
    else:
        segments = np.array([[args.start_ms, args.end_ms]], dtype=np.int32)

    embedder = AudioEmbedder(model_dir=args.model)
    vector = embedder.get_embedding(
        wav_fn=args.wav, segments=segments, trim_sil_thres=args.trim_sil_thres
    )
    print(vector)


if __name__ == "__main__":
    main()
