#!/usr/bin/env python3

import unittest

from acn_embed.embed.data.microbatch import span_info
from acn_embed.embed.data.microbatch.span_info import Span


class TestSpanInfo(unittest.TestCase):
    def test_span_info(self):
        sp_info = span_info.SpanInfo(utt_len=6)
        sp_info.dump()
        self.assertEqual(len(sp_info.idx_to_span), 21)
        self.assertEqual(sp_info.idx_to_span[0], Span(0, 1))
        self.assertEqual(sp_info.idx_to_span[4], Span(0, 5))
        self.assertEqual(sp_info.idx_to_span[7], Span(1, 3))
        self.assertEqual(sp_info.idx_to_span[12], Span(2, 4))
        self.assertEqual(sp_info.idx_to_span[15], Span(3, 4))
        self.assertEqual(sp_info.idx_to_span[18], Span(4, 5))
        self.assertEqual(sp_info.idx_to_span[20], Span(5, 6))


if __name__ == "__main__":
    unittest.main()
