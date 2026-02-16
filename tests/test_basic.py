import unittest
import numpy as np
from labeltool import segmentation

class TestSegmentation(unittest.TestCase):
    def test_basic_import(self):
        # Basic test to ensure the package can be imported and logic accessed
        self.assertTrue(hasattr(segmentation, 'run_multi_process'))

if __name__ == "__main__":
    unittest.main()
