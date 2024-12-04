#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from acn_embed.util.data.storage import get_feat_dim, H5FeatStoreReader, H5FeatStoreWriter


class StorageTest(unittest.TestCase):

    def test_h5(self):
        _, tmpf = tempfile.mkstemp(suffix=".h5")
        tmpf = Path(tmpf)
        writer = H5FeatStoreWriter(tmpf, compress=True)
        a = np.random.rand(3, 9, 4)
        b = np.random.rand(10, 11, 12)
        c = np.random.rand(13, 14, 15)
        writer.add_nparr(0, a)
        writer.add_nparr(3000, b)
        writer.add_nparr(3000000, c)
        self.assertEqual(writer.num_written, 3)
        writer.close()

        self.assertEqual(get_feat_dim(tmpf), 4)

        reader = H5FeatStoreReader(tmpf)
        np.testing.assert_almost_equal(reader.get_nparr(3000), b)
        np.testing.assert_almost_equal(reader.get_nparr(0), a)
        np.testing.assert_almost_equal(reader.get_nparr(3000000), c)

        # Note that "size" can be deceptive if the indices are not contiguous
        self.assertEqual(reader.size, 3000001)

        self.assertRaises(BaseException, reader.get_nparr, 5)
        self.assertRaises(BaseException, reader.get_nparr, 5000000)


if __name__ == "__main__":
    unittest.main()
