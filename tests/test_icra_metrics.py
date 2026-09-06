import unittest

import numpy as np

from experiments.icra.metrics import evaluate_instances


def square(y1, y2, x1, x2):
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


class InstanceMetricTests(unittest.TestCase):
    def test_hungarian_matching_ignores_instance_names(self):
        gt = {
            "left": {1: square(1, 5, 1, 5)},
            "right": {1: square(7, 11, 7, 11)},
        }
        pred = {
            "prediction_a": {1: square(7, 11, 7, 11)},
            "prediction_b": {1: square(1, 5, 1, 5)},
        }
        result = evaluate_instances(pred, gt, [1])
        self.assertAlmostEqual(result.scene_instance_miou, 1.0)
        self.assertAlmostEqual(result.foreground_iou, 1.0)
        self.assertAlmostEqual(result.boundary_f1, 1.0)
        self.assertAlmostEqual(result.failure_rate, 0.0)

    def test_unmatched_ground_truth_is_zero_not_dropped(self):
        gt = {
            "left": {1: square(1, 5, 1, 5)},
            "right": {1: square(7, 11, 7, 11)},
        }
        pred = {"only_one": {1: square(1, 5, 1, 5)}}
        result = evaluate_instances(pred, gt, [1])
        self.assertAlmostEqual(result.scene_instance_miou, 0.5)
        self.assertAlmostEqual(result.failure_rate, 0.5)

    def test_no_predictions_is_full_failure(self):
        gt = {"object": {1: square(2, 8, 2, 8)}}
        result = evaluate_instances({}, gt, [1])
        self.assertEqual(result.scene_instance_miou, 0.0)
        self.assertEqual(result.foreground_iou, 0.0)
        self.assertEqual(result.failure_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

