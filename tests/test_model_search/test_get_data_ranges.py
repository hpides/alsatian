import unittest

from model_search.approaches.shift import get_data_ranges


class TestGetDataRanges(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_35_820(self):
        search_space_len = 35
        data_len = 820
        result = get_data_ranges(search_space_len, data_len)
        counts = [x[1] - x[0] for x in result]
        # check that count always doubles
        for i in range(len(counts) - 2):
            self.assertEqual(counts[i] * 2, counts[i + 1])
        # check that last counts sum up to data len
        self.assertEqual(sum(counts), data_len)
        self.assertTrue(counts[-1] > counts[-2])

    def test_35_800(self):
        search_space_len = 35
        data_len = 800
        result = get_data_ranges(search_space_len, data_len)
        counts = [x[1] - x[0] for x in result]
        # check that count always doubles
        for i in range(len(counts) - 2):
            self.assertEqual(counts[i] * 2, counts[i + 1])
        # check that last counts sum up to data len
        self.assertEqual(sum(counts), data_len)
        self.assertTrue(counts[-1] > counts[-2])

    def test_100_800(self):
        search_space_len = 35
        data_len = 800
        result = get_data_ranges(search_space_len, data_len)
        counts = [x[1] - x[0] for x in result]
        # check that count always doubles
        for i in range(len(counts) - 2):
            self.assertEqual(counts[i] * 2, counts[i + 1])
        # check that last counts sum up to data len
        self.assertEqual(sum(counts), data_len)
        self.assertTrue(counts[-1] > counts[-2])


    def test_35_100(self):
        data_len = 100
        search_space_len = 35
        result = get_data_ranges(search_space_len, data_len)
        counts = [x[1] - x[0] for x in result]
        # check that count always doubles
        for i in range(len(counts) - 2):
            self.assertEqual(counts[i] * 2, counts[i + 1])
        # check that last counts sum up to data len
        self.assertEqual(sum(counts), data_len)
        self.assertTrue(counts[-1] > counts[-2])

