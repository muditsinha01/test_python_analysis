import unittest
import json
import os
import shutil
from data_prep import load_json_annotations, enhance_annotations_with_negatives, EnhancedAnnotations, Annotation

import unittest
import json
import os
import shutil
from data_prep import load_json_annotations, enhance_annotations_with_negatives, EnhancedAnnotations, Annotation

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        # Create a sample annotations dictionary
        self.sample_annotations = {
            "file1.c": {
                "5": {"char_ranges": [[10, 20]]}
            }
        }
        # Create a temporary annotations file
        with open('temp_annotations.json', 'w') as f:
            json.dump(self.sample_annotations, f)
        
        # Create a temporary dataset directory with a sample file
        os.makedirs('./temp_dataset', exist_ok=True)
        with open('./temp_dataset/file1.c', 'w') as f:
            f.write('\n'.join(['line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7']))

    def test_load_json_annotations(self):
        annotations = load_json_annotations('temp_annotations.json')
        self.assertEqual(annotations, self.sample_annotations)

    def test_enhance_annotations_with_negatives(self):
        annotations = {k: {line: Annotation(**v) for line, v in file_annot.items()} 
                       for k, file_annot in self.sample_annotations.items()}
        enhanced_annotations = enhance_annotations_with_negatives(annotations, './temp_dataset')
        
        self.assertIsInstance(enhanced_annotations, EnhancedAnnotations)
        self.assertIn('file1.c', enhanced_annotations.annotations)
        self.assertIn('5', enhanced_annotations.annotations['file1.c'])
        self.assertEqual(enhanced_annotations.annotations['file1.c']['5'].is_vulnerable, 1)
        
        # Check if a negative sample was added
        negative_samples = [sample for sample in enhanced_annotations.annotations['file1.c'].values() if sample.is_vulnerable == 0]
        self.assertGreater(len(negative_samples), 0)

    def tearDown(self):
        # Clean up temporary files
        os.remove('temp_annotations.json')
        shutil.rmtree('./temp_dataset')

if __name__ == '__main__':
    unittest.main()