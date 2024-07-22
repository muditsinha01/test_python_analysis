import unittest
import pandas as pd
import numpy as np
import gzip
import os
from simple_preprocesing import *

class TestSimplePreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a small sample dataset for testing
        cls.sample_data = pd.DataFrame({
            'Source code': ['def hello():\n    print("Hello")', 'x = 10\nprint(x)'],
            'Vulnerability type': ['Buffer Overflow', 'SQL Injection']
        })

        # Save sample data to a gzip file
        cls.test_file_path = 'test_FormAI_dataset.csv.gz'
        with gzip.open(cls.test_file_path, 'wt') as f:
            cls.sample_data.to_csv(f, index=False)

    @classmethod
    def tearDownClass(cls):
        # Remove the test file after all tests are done
        if os.path.exists(cls.test_file_path):
            os.remove(cls.test_file_path)

    def test_load_and_inspect_data(self):
        dataset = load_and_inspect_data(self.test_file_path)
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertEqual(len(dataset), 2)
        self.assertIn('Source code', dataset.columns)
        self.assertIn('Vulnerability type', dataset.columns)

    def test_clean_data(self):
        dataset = load_and_inspect_data(self.test_file_path)
        cleaned_data = clean_data(dataset)
        self.assertEqual(len(cleaned_data), len(dataset))  # Assuming no duplicates or NaNs in sample data

    def test_preprocess_data(self):
        dataset = load_and_inspect_data(self.test_file_path)
        cleaned_data = clean_data(dataset)
        preprocessed_data, tokenizer = preprocess_data(cleaned_data)
        self.assertIn('Code_Tokens', preprocessed_data.columns)
        self.assertIn('Code_Tokens_Padded', preprocessed_data.columns)
        self.assertIn('Vulnerability_OHE', preprocessed_data.columns)

    def test_build_model(self):
        dataset = load_and_inspect_data(self.test_file_path)
        cleaned_data = clean_data(dataset)
        preprocessed_data, tokenizer = preprocess_data(cleaned_data)
        max_length = max([len(seq) for seq in preprocessed_data['Code_Tokens']])
        model = build_model(tokenizer, max_length, output_dim=len(preprocessed_data['Vulnerability_OHE'][0]))
        self.assertIsNotNone(model)

    def test_train_and_evaluate_model(self):
        dataset = load_and_inspect_data(self.test_file_path)
        cleaned_data = clean_data(dataset)
        preprocessed_data, tokenizer = preprocess_data(cleaned_data)
        max_length = max([len(seq) for seq in preprocessed_data['Code_Tokens']])
        model = build_model(tokenizer, max_length, output_dim=len(preprocessed_data['Vulnerability_OHE'][0]))
        
        # This test might take a while to run, so you might want to reduce epochs or skip it in quick tests
        train_and_evaluate_model(model, preprocessed_data)
        # Since we can't easily assert the accuracy, we're just checking that it runs without errors

    def test_full_pipeline(self):
        # Test the entire pipeline from loading data to training the model
        dataset = load_and_inspect_data(self.test_file_path)
        cleaned_data = clean_data(dataset)
        preprocessed_data, tokenizer = preprocess_data(cleaned_data)
        max_length = max([len(seq) for seq in preprocessed_data['Code_Tokens']])
        model = build_model(tokenizer, max_length, output_dim=len(preprocessed_data['Vulnerability_OHE'][0]))
        train_and_evaluate_model(model, preprocessed_data)
        # The test passes if no exceptions are raised during the pipeline execution

if __name__ == '__main__':
    unittest.main()