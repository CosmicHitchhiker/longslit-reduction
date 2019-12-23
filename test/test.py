import unittest

import longslit.pipeline as ppl


class ReadConfigTestCase(unittest.TestCase):
    def test_mode(self):
        name = "empty.json";
        cfg = ppl.read_config(name)
        self.assertTrue(type(cfg[0]) == type(dict()),
                        msg='mode setting should be dictionary')
    def test_margins(self):
        name = "empty.json";
        cfg = ppl.read_config(name)
        margins = np.array(cfg[1])
        self.assertTrue(margins.dtype == 'int64',
                        msg='mode should be int')