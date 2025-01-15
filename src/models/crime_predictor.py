import os
import pandas as pd
import joblib
import logging
from typing import Dict, Any

class CrimePredictor:
    def __init__(self, model_dir: str) -> None:
        try:
            self.model = joblib.load(os.path.join(model_dir, 'crime_prediction_model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
            
            self.feature_order = ['TIME OCC', 'AREA', 'Rpt Dist No', 'Part 1-2', 
                                'Crm Cd', 'Vict Age', 'LAT', 'LON']
            logging.info("CrimePredictor initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize CrimePredictor: {str(e)}")
            raise

    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Validate input
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            if input_data.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Check required columns
            missing_cols = set(self.feature_order) - set(input_data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Process data
            X = input_data[self.feature_order].copy()
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return {
                'predicted_category': self.label_encoder.inverse_transform(prediction)[0],
                'confidence': float(max(probabilities[0]))
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Return the importance of each feature in the model."""
        try:
            importance = self.model.feature_importances_
            return dict(zip(self.feature_order, importance))
        except AttributeError:
            logging.warning("Model does not support feature importance")
            return {}

    def explain_prediction(self, input_data: pd.DataFrame, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed explanation of prediction."""
        try:
            # Get feature importance
            importance = self.get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get input values for top features
            feature_values = {
                feature: input_data[feature].iloc[0] 
                for feature, _ in top_features
            }
            
            return {
                'prediction': prediction['predicted_category'],
                'confidence': prediction['confidence'],
                'top_features': feature_values,
                'interpretation': (
                    f"Predicted {prediction['predicted_category']} "
                    f"with {prediction['confidence']:.2%} confidence. "
                    f"Top contributing features: {', '.join(feature_values.keys())}"
                )
            }
            
        except Exception as e:
            logging.error(f"Explanation failed: {str(e)}")
            raise
