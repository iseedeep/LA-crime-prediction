import unittest
import pandas as pd
import numpy as np
from src.models.enhanced_predictor import EnhancedCrimePredictor
import os
import shutil

class TestEnhancedCrimePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment and sample data"""
        cls.test_dir = 'enhanced_models_test'
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.predictor = EnhancedCrimePredictor(model_dir=cls.test_dir)
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        cls.train_data = pd.DataFrame({
            'TIME OCC': np.random.randint(0, 2400, n_samples),
            'AREA': np.random.randint(1, 15, n_samples),
            'Rpt Dist No': np.random.randint(100, 999, n_samples),
            'Part 1-2': np.random.randint(1, 3, n_samples),
            'Crm Cd': np.random.randint(600, 900, n_samples),
            'Vict Age': np.random.randint(18, 80, n_samples),
            'LAT': 34.0522 + np.random.normal(0, 0.1, n_samples),
            'LON': -118.2437 + np.random.normal(0, 0.1, n_samples),
            'Target': np.random.randint(0, 4, n_samples)  # 4 crime categories
        })
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test directory after tests"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up test case with sample prediction data"""
        self.test_data = pd.DataFrame([{
            'TIME OCC': 1200,
            'AREA': 1,
            'Rpt Dist No': 123,
            'Part 1-2': 1,
            'Crm Cd': 624,
            'Vict Age': 30,
            'LAT': 34.0522,
            'LON': -118.2437
        }])

    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        processed_data = self.predictor.engineer_features(self.test_data)
        
        expected_features = [
            'hour', 'is_night', 'is_rush_hour',
            'dist_from_center', 'location_cluster',
            'time_area_interaction', 'age_time_interaction'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, processed_data.columns)
    
    def test_model_training(self):
        """Test model training pipeline"""
        try:
            results = self.predictor.train(self.train_data)
            
            # Check if all expected components are present
            self.assertIn('model', results)
            self.assertIn('scaler', results)
            self.assertIn('feature_cols', results)
            self.assertIn('classification_report', results)
            
            # Check if model files were created
            expected_files = [
                'stacking_classifier.joblib',
                'enhanced_scaler.joblib',
                'confusion_matrix.png',
                'roc_curves.png'
            ]
            
            for file in expected_files:
                self.assertTrue(
                    os.path.exists(os.path.join(self.test_dir, file)),
                    f"Missing file: {file}"
                )
                
        except Exception as e:
            self.fail(f"Model training failed: {str(e)}")
    
    def test_prediction_output(self):
        """Test prediction functionality"""
        # First train the model
        self.predictor.train(self.train_data)
        
        # Make prediction
        result = self.predictor.predict(self.test_data)
        
        # Check prediction structure
        self.assertIn('predicted_category', result)
        self.assertIn('confidence', result)
        self.assertIn('feature_importance', result)
        
        # Check data types
        self.assertIsInstance(result['predicted_category'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['feature_importance'], dict)
        
        # Check confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test missing columns
        incomplete_data = pd.DataFrame({
            'TIME OCC': [1200],
            'AREA': [1]
        })
        
        with self.assertRaises(ValueError):
            self.predictor.predict(incomplete_data)
        
        # Test invalid values
        invalid_data = self.test_data.copy()
        invalid_data.loc[0, 'TIME OCC'] = 2500  # Invalid time
        
        with self.assertRaises(ValueError):
            self.predictor.predict(invalid_data)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train and save model
        self.predictor.train(self.train_data)
        
        # Create new predictor instance
        new_predictor = EnhancedCrimePredictor(model_dir=self.test_dir)
        
        # Make predictions with both predictors
        orig_pred = self.predictor.predict(self.test_data)
        new_pred = new_predictor.predict(self.test_data)
        
        # Check predictions match
        self.assertEqual(
            orig_pred['predicted_category'],
            new_pred['predicted_category']
        )
        self.assertAlmostEqual(
            orig_pred['confidence'],
            new_pred['confidence'],
            places=5
        )

if __name__ == '__main__':
    unittest.main()