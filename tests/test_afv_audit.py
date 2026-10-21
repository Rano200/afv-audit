"""Mechanics tests only (transform inversion, window labelling). Synthetic numbers here support no claim about any AFV.
Run:  python -m unittest discover -s tests   (from the afv_audit folder)"""
import os
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import afv_audit as T  # noqa: E402

# a made-up 2026 -> 2023 transform: scale 0.25, rotate 90 degrees about z, translate
M = np.array([[0.0, -0.25, 0.0, 100.0],
              [0.25, 0.0, 0.0, 50.0],
              [0.0, 0.0, 0.25, 10.0]])
A = np.vstack([M, [0, 0, 0, 1]])
AINV = np.linalg.inv(A)


class TransformTest(unittest.TestCase):
    def test_three_point_inversion(self):
        p26 = np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 800.0, 40.0]])
        p23 = T.apply_h(A, p26)
        # worked by hand: (x,y,z) -> (-0.25y+100, 0.25x+50, 0.25z+10)
        np.testing.assert_allclose(p23, [[100, 50, 10], [100, 150, 10], [-100, 50, 20]])
        np.testing.assert_allclose(T.apply_h(AINV, p23), p26, atol=1e-9)

    def test_nml_nodes_map_to_2026(self):
        # NML nodes are in the 2023 frame; 2026 = inverse transform
        nodes23 = np.array([[100.0, 50.0, 10.0], [100.0, 150.0, 10.0], [-100.0, 50.0, 20.0]])
        np.testing.assert_allclose(T.apply_h(AINV, nodes23), [[0, 0, 0], [400, 0, 0], [0, 800, 40]], atol=1e-9)

    def test_cube_box_contains_its_corner(self):
        z, lo23, hi23, lo, hi = T.cube_geometry("fibers_s1a_00497z_01497y_03997x_256_v00", A, AINV)
        self.assertEqual(z, 497)
        np.testing.assert_allclose(lo23, [3997, 1497, 497])
        np.testing.assert_allclose(hi23 - lo23, [256, 256, 256])
        self.assertTrue(np.all(lo < hi))


class WindowTest(unittest.TestCase):
    def test_three_point_run_is_one_window(self):
        run = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 0, 0]])  # arc length exactly 200
        w = T.cut_windows(run)
        self.assertEqual(len(w), 1)
        self.assertEqual(len(w[0]), 3)

    def test_short_run_has_no_window(self):
        self.assertEqual(T.cut_windows(np.array([[0.0, 0, 0], [100, 0, 0], [199, 0, 0]])), [])

    def test_labelling_three_points(self):
        hand = T.densify(np.array([[0.0, 0, 0], [200.0, 0, 0]]))
        tree = cKDTree(hand)
        # offsets from the hand fibre: 0, 0, 30 voxels -> 2 of 3 points within 20
        w = np.array([[0.0, 0, 0], [100.0, 0, 0], [200.0, 30, 0]])
        d, _ = tree.query(w)
        f = float((d <= 20).mean())
        self.assertAlmostEqual(f, 2 / 3)
        self.assertEqual(T.label_from_fraction(f), "MID")
        self.assertEqual(T.label_from_fraction(1.0), "GOOD")
        self.assertEqual(T.label_from_fraction(0.8), "GOOD")
        self.assertEqual(T.label_from_fraction(0.2), "BAD")
        self.assertEqual(T.label_from_fraction(0.0), "BAD")
        # null: hand fibre moved 200 voxels in XY -> window all far
        dn, _ = cKDTree(hand + np.array([0.0, 200.0, 0.0])).query(w)
        self.assertEqual(T.label_from_fraction(float((dn <= 20).mean())), "BAD")


class LinksMechanicsTest(unittest.TestCase):
    """Synthetic greedy-cover / union-find mechanics for --links. No claim about any AFV."""

    def _line(self, x0, x1):
        return np.c_[np.arange(x0, x1 + 1, dtype=float), np.zeros(x1 - x0 + 1), np.zeros(x1 - x0 + 1)]

    def test_union_find(self):
        uf = T.UF()
        for a, b in [(1, 2), (3, 4), (2, 3)]:
            uf.union(a, b)
        uf.find(9)
        self.assertEqual(len({uf.find(i) for i in (1, 2, 3, 4)}), 1)
        self.assertNotEqual(uf.find(9), uf.find(1))

    def test_greedy_cover_pieces_vs_chains(self):
        R = self._line(0, 99)
        w = np.ones(len(R))
        groups = {1: self._line(0, 49), 2: self._line(50, 99), 3: self._line(0, 10), 4: self._line(0, 99) + [0, 500, 0]}
        # fibers 1 and 2 each cover half; 3 is a subset of 1; 4 is far away (never near)
        self.assertAlmostEqual(T.union_cover(R, w, groups), 1.0)
        self.assertEqual(T.greedy_cover(R, w, groups, 0.9), 2)
        # linking 1 and 2 into one chain -> one chain covers the stretch
        uf = T.UF()
        uf.union(1, 2)
        ch = {}
        for f, p in groups.items():
            ch.setdefault(uf.find(f), []).append(p)
        chain_groups = {k: np.concatenate(v) for k, v in ch.items()}
        self.assertEqual(T.greedy_cover(R, w, chain_groups, 0.9), 1)
        # unreachable target -> None
        self.assertIsNone(T.greedy_cover(R, w, {3: groups[3]}, 0.9))
        self.assertEqual(T.greedy_cover(R, w, {3: groups[3]}, 0.05), 1)


if __name__ == "__main__":
    unittest.main()
