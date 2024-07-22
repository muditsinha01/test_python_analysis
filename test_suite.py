import unittest
from test_simple_preprocessing import TestSimplePreprocessing
from test_dataset_analyzer import TestDatasetAnalyzer
from test_data_prep import TestDataPrep

def create_test_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestSimplePreprocessing))
    test_suite.addTest(unittest.makeSuite(TestDatasetAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestDataPrep))
    return test_suite

if __name__ == '__main__':
    suite = create_test_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)