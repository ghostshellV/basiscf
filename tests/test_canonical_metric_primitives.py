import unittest

import numpy as np
import torch

from src.evaluation.metrics import (
    DEFAULT_MAD_INV_CAP,
    actionability_np,
    in_bounds_np,
    proximity_l2_per_elem_np,
    proximity_l2_per_elem_torch,
    proximity_mad_per_elem_np,
    proximity_mad_per_elem_torch,
    smoothness_per_elem_np,
    smoothness_per_elem_torch,
    sparsity_l1_per_elem_np,
    sparsity_l1_per_elem_torch,
)


class CanonicalMetricPrimitiveParityTests(unittest.TestCase):
    def setUp(self) -> None:
        # Shape: (N, T, D) for torch, and (T, D) for numpy per-sample primitives.
        self.delta_torch = torch.tensor(
            [
                [[0.10, -0.20, 0.05], [0.20, -0.10, 0.10], [0.10, -0.05, -0.10], [0.05, 0.00, -0.15]]
            ],
            dtype=torch.float32,
        )
        self.delta_np = self.delta_torch[0].numpy()
        self.mad_inv_np = np.array([1.0, 200.0, 0.5], dtype=np.float32)
        self.mad_inv_torch = torch.tensor(self.mad_inv_np)

    def test_proximity_l2_numpy_torch_parity(self) -> None:
        np_val = proximity_l2_per_elem_np(self.delta_np)
        torch_val = float(proximity_l2_per_elem_torch(self.delta_torch).item())
        self.assertAlmostEqual(np_val, torch_val, places=7)

    def test_proximity_mad_numpy_torch_parity_with_cap(self) -> None:
        np_val = proximity_mad_per_elem_np(
            self.delta_np,
            mad_inv=self.mad_inv_np,
            mad_inv_cap=DEFAULT_MAD_INV_CAP,
        )
        torch_val = float(
            proximity_mad_per_elem_torch(
                self.delta_torch,
                mad_inv=self.mad_inv_torch,
                mad_inv_cap=DEFAULT_MAD_INV_CAP,
            ).item()
        )
        self.assertAlmostEqual(np_val, torch_val, places=6)

    def test_sparsity_numpy_torch_parity(self) -> None:
        np_val = sparsity_l1_per_elem_np(self.delta_np)
        torch_val = float(sparsity_l1_per_elem_torch(self.delta_torch).item())
        self.assertAlmostEqual(np_val, torch_val, places=7)

    def test_smoothness_numpy_torch_parity(self) -> None:
        np_val = smoothness_per_elem_np(self.delta_np)
        torch_val = float(smoothness_per_elem_torch(self.delta_torch).item())
        self.assertAlmostEqual(np_val, torch_val, places=7)

    def test_actionability_all_mutable_is_trivially_true(self) -> None:
        mutable_mask = np.ones(self.delta_np.shape[1], dtype=bool)
        self.assertEqual(actionability_np(self.delta_np, mutable_mask), 1.0)

    def test_actionability_detects_immutable_change(self) -> None:
        mutable_mask = np.array([True, False, True], dtype=bool)
        # delta changes feature index 1, which is immutable in this mask.
        self.assertEqual(actionability_np(self.delta_np, mutable_mask), 0.0)

    def test_in_bounds(self) -> None:
        cf_good = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
        cf_bad = np.array([[0.0, 1.2], [0.8, 1.0]], dtype=np.float32)
        self.assertEqual(in_bounds_np(cf_good), 1.0)
        self.assertEqual(in_bounds_np(cf_bad), 0.0)


if __name__ == "__main__":
    unittest.main()
