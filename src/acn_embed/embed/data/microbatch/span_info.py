import scipy.special


class Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end


class SpanInfo:
    def __init__(self, utt_len):
        self.utt_len = utt_len
        self.idx_to_span = []
        for start_idx in range(utt_len):
            for span_len in range(1, utt_len - start_idx + 1):
                span = Span(start=start_idx, end=start_idx + span_len)
                self.idx_to_span.append(span)
        assert len(self.idx_to_span) == scipy.special.comb(utt_len + 1, 2)

    def dump(self):
        for idx, span in enumerate(self.idx_to_span):
            print(idx, span.start, span.end)
