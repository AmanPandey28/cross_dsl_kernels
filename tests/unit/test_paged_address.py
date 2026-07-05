from __future__ import annotations

import unittest

from crossdsl_kernels.references.paged_address import (
    address_for_token,
    hnd_linear_index,
    nhd_linear_index,
)


class PagedAddressTests(unittest.TestCase):
    def test_page_boundary(self) -> None:
        table = [7, 3, 5]
        self.assertEqual(address_for_token(15, 16, table).physical_page, 7)
        self.assertEqual(address_for_token(15, 16, table).offset, 15)
        self.assertEqual(address_for_token(16, 16, table).logical_page, 1)
        self.assertEqual(address_for_token(16, 16, table).physical_page, 3)
        self.assertEqual(address_for_token(17, 16, table).offset, 1)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            address_for_token(0, 0, [1])
        with self.assertRaises(ValueError):
            address_for_token(-1, 16, [1])
        with self.assertRaises(IndexError):
            address_for_token(32, 16, [1])
        with self.assertRaises(ValueError):
            address_for_token(0, 16, [-1])

    def test_layout_flattening(self) -> None:
        self.assertEqual(
            nhd_linear_index(2, 3, 1, 5, page_size=16, num_heads=4, head_dim=8),
            (((2 * 16 + 3) * 4 + 1) * 8) + 5,
        )
        self.assertEqual(
            hnd_linear_index(2, 3, 1, 5, page_size=16, num_heads=4, head_dim=8),
            (((2 * 4 + 1) * 16 + 3) * 8) + 5,
        )


if __name__ == "__main__":
    unittest.main()
