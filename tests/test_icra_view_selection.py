import unittest

import numpy as np

from experiments.icra.view_selection import geometry_order, select_views, uniform_exact_budget


class ViewSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.asarray([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16,
                               22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36])
        angles = np.linspace(0, 2 * np.pi, len(self.ids), endpoint=False)
        poses = np.repeat(np.eye(4)[None], len(self.ids), axis=0)
        poses[:, 0, 3] = np.cos(angles)
        poses[:, 1, 3] = np.sin(angles)
        self.poses = poses[:, :3, :4]

    def test_exact_budgets_are_nested(self):
        order = np.arange(24)
        four = set(uniform_exact_budget(order, 4))
        eight = set(uniform_exact_budget(order, 8))
        sixteen = set(uniform_exact_budget(order, 16))
        self.assertEqual(len(four), 4)
        self.assertEqual(len(eight), 8)
        self.assertEqual(len(sixteen), 16)
        self.assertLessEqual(four, eight)
        self.assertLessEqual(eight, sixteen)

    def test_random_is_reproducible_and_seeded(self):
        first = select_views("random", self.ids, self.poses, 8, selection_seed=3)
        repeat = select_views("random", self.ids, self.poses, 8, selection_seed=3)
        other = select_views("random", self.ids, self.poses, 8, selection_seed=4)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first.selected_camera_ids, other.selected_camera_ids)

    def test_geometry_order_is_a_deterministic_permutation(self):
        first = geometry_order(self.poses, self.ids)
        second = geometry_order(self.poses, self.ids)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), set(self.ids))


if __name__ == "__main__":
    unittest.main()

