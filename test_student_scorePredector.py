import importlib.util
import unittest
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).with_name('student_scorePredector.py')
spec = importlib.util.spec_from_file_location('student_scorePredector', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExamPredictionTests(unittest.TestCase):
    def test_example_student_prediction_is_finite_and_within_score_range(self):
        self.assertTrue(hasattr(module, 'predict_score'), 'predict_score function is missing')
        student = {'hours_studied': 6.5, 'attendance_rate': 85, 'midterm_score': 80}
        prediction = module.predict_score(module.train_model(module.load_data()), student)
        self.assertTrue(0 <= prediction <= 100)
        self.assertTrue(pd.notna(prediction))

    def test_validation_rejects_impossible_student(self):
        self.assertTrue(hasattr(module, 'predict_score'), 'predict_score function is missing')
        with self.assertRaises(ValueError):
            module.predict_score(module.train_model(module.load_data()),
                                 {'hours_studied': -1, 'attendance_rate': 101, 'midterm_score': 80})

    def test_evaluation_has_one_out_of_sample_prediction_per_student(self):
        self.assertTrue(hasattr(module, 'evaluate_model'), 'evaluate_model function is missing')
        result = module.evaluate_model(module.load_data())
        self.assertEqual(len(result['predictions']), len(module.load_data()))
        self.assertGreaterEqual(result['mae'], 0)


if __name__ == '__main__':
    unittest.main()
