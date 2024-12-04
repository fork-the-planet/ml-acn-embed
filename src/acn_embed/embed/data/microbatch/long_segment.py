class LongSegment:
    def __init__(self, num, token_list, utt_id, ms_offset):
        self.num = num
        self.token_list = token_list
        self.utt_id = utt_id
        self.ms_offset = ms_offset
        self.span_idx_to_pron = {}

    def __eq__(self, other):
        return (
            self.num == other.num
            and self.token_list == other.token_list
            and self.utt_id == other.utt_id
            and self.ms_offset == other.ms_offset
        )

    def __str__(self):
        return f"LongSegment {self.num} {self.utt_id} ms_offset={self.ms_offset} {self.token_list}"

    @property
    def length(self):
        return len(self.token_list)

    def add_span_indexing(self, max_audio_len_frame, max_pron_len):
        """
        Adds this attribute:
            span_idx_to_pron: maps span_idx -> pron
        """
        pron_list = [token["pron"] for token in self.token_list]
        span_idx = 0
        for start_idx in range(self.length):
            for end_idx in range(start_idx + 1, self.length + 1):
                num_frames = (
                    self.token_list[end_idx - 1]["end_frame"]
                    - self.token_list[start_idx]["start_frame"]
                )
                pron = " ".join(pron_list[start_idx:end_idx])
                if num_frames > max_audio_len_frame or len(pron.split()) > max_pron_len:
                    self.span_idx_to_pron[span_idx] = None
                else:
                    self.span_idx_to_pron[span_idx] = pron
                span_idx += 1


class LongSegSpan:
    def __init__(self, long_seg_num, span_idx):
        self.long_seg_num = long_seg_num
        self.span_idx = span_idx

    def __eq__(self, other):
        return self.long_seg_num == other.long_seg_num and self.span_idx == other.span_idx

    def __str__(self):
        return f"LSS num={self.long_seg_num=} span_idx={self.span_idx=}"
