"""Remote synthetic regression checks for coverage and unequal capture sizes."""

import copy
import unittest

from dust.grouping import CAPTURE_POLICY
from dust.splitting import SPLIT_POLICY, allocate_splits, split_quality
from freeze_baseline import GEOMETRY, check_baseline_manifest


def record(group, sensor, keep=10000, dust=1000, car=0):
    semantic = [0] * 29
    semantic[1], semantic[9], semantic[12] = keep - car, dust, car
    return {"group": group, "sensor": sensor, "class_counts": [keep, dust],
            "finite_semantic_counts": semantic}


class SplitTests(unittest.TestCase):
    def unequal_captures(self):
        rows = []
        for gi, nframes in enumerate((600, 100, 100, 50, 50, 50, 50)):
            for sensor in ("ls64", "ouster"):
                # Ouster needs one of three substantial dust captures in each fold.
                dust = 1000 if sensor == 'ls64' or gi in (3, 4, 5) else 1
                car = 300 if sensor == 'ls64' and gi in (0, 1, 2) else 0
                rows.extend(record(f"capture_{gi}", sensor, dust=dust, car=car) for _ in range(nframes))
        return rows

    def test_frame_balance_binary_mass_and_car_coverage_with_indivisible_groups(self):
        rows = self.unequal_captures()
        original = copy.deepcopy(rows)
        assigned, report = allocate_splits(rows, 42, (.7, .15, .15))
        self.assertEqual(rows, original)
        self.assertGreater(report['splits']['train']['frame_fraction'], .6)
        self.assertLess(report['splits']['train']['frame_fraction'], .8)
        for split in ('train', 'val', 'test'):
            sensors = report['splits'][split]['sensors']
            self.assertGreaterEqual(sensors['ls64']['car_points'], 100)
            self.assertGreaterEqual(sensors['ouster']['dust_points'], 1000)
        again, _ = allocate_splits(list(reversed(rows)), 42, (.7, .15, .15))
        self.assertEqual(assigned, again)
        self.assertEqual(set(assigned.values()), {'train', 'val', 'test'})

    def test_sparse_car_captures_are_reported_without_inventing_coverage(self):
        rows = [record(f'g{i}', 'sensor', car=20 if i < 2 else 0) for i in range(4)]
        _, report = allocate_splits(rows, 42, (.7, .15, .15))
        self.assertEqual(report['limited_car_coverage'][0]['independent_groups_with_points'], 2)
        self.assertFalse(any(c['category'] == 'Car' for c in report['constraints']))
        self.assertTrue(report['coverage_passed'])

    def test_insufficient_or_jointly_incompatible_captures_fail_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient independent'):
            allocate_splits([record(f'g{i}', 'sensor') for i in range(2)], 42, (.7, .15, .15))
        # Three pairs must each occupy all folds, which four groups cannot satisfy.
        rows = []
        for si, indices in enumerate(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))):
            rows.extend(record(f'g{i}', f's{si}') for i in indices)
        with self.assertRaisesRegex(ValueError, 'No fallback'):
            allocate_splits(rows, 42, (.7, .15, .15))

    def test_quality_and_freeze_reject_coverage_loss_and_old_protocol(self):
        rows = [record(f'g{i}', 'sensor', car=100) for i in range(3)]
        assigned, _ = allocate_splits(rows, 42, (.7, .15, .15))
        prepared = [dict(r, split=assigned[r['group']]) for r in rows]
        manifest = {'preparation_protocol': CAPTURE_POLICY, 'split_protocol': SPLIT_POLICY,
                    'geometry': GEOMETRY, 'split_seed': 42, 'requested_split_ratios': [.7, .15, .15],
                    'frames': prepared}
        check_baseline_manifest(manifest)
        old = copy.deepcopy(manifest)
        old.pop('split_protocol')
        with self.assertRaisesRegex(ValueError, 'coverage-balanced v3'):
            check_baseline_manifest(old)
        broken = copy.deepcopy(prepared)
        for r in broken:
            if r['split'] == 'test':
                r['class_counts'][1] = r['finite_semantic_counts'][9] = 0
        with self.assertRaisesRegex(ValueError, 'Insufficient independent'):
            split_quality(broken)


if __name__ == '__main__':
    unittest.main()
