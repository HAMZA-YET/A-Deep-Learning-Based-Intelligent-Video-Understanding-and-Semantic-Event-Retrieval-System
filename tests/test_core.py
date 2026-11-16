import unittest
from scenemind.core import Config, windows, sample_times, select_results

class CoreTests(unittest.TestCase):
    def test_short_video(self):
        self.assertEqual(windows(2, Config()), [(0, 2)])

    def test_tail_coverage(self):
        self.assertEqual(windows(13, Config()), [(0, 8), (4, 12), (8, 13)])

    def test_exact_boundary(self):
        self.assertEqual(windows(12, Config()), [(0, 8), (4, 12)])

    def test_bad_duration(self):
        for duration in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                windows(duration, Config())

    def test_invalid_config(self):
        for kwargs in ({'stride_seconds': 9}, {'frames_per_window': 0}, {'batch_size': 0}, {'window_seconds': 0}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_midpoint_sampling(self):
        self.assertEqual(sample_times(0, 8, 4), [1, 3, 5, 7])

    def test_invalid_samples(self):
        with self.assertRaises(ValueError):
            sample_times(2, 1, 8)

    def test_rejection(self):
        self.assertEqual(select_results([{'start': 0, 'end': 8, 'score': .1}], 5, .25), [])

    def test_ranking_dedup_and_topk(self):
        rows = [{'start': 1, 'end': 9, 'score': .8},
                {'start': 20, 'end': 28, 'score': .7},
                {'start': 0, 'end': 8, 'score': .9}]
        result = select_results(rows, 2, .2)
        self.assertEqual([r['score'] for r in result], [.9, .7])

    def test_invalid_search(self):
        with self.assertRaises(ValueError):
            select_results([], 0, .2)

if __name__ == '__main__':
    unittest.main()
