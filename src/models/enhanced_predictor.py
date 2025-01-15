import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import logging
from datetime import datetime

class EnhancedCrimePredictor:
    def __init__(self, model_dir='enhanced_models'):
        self.model_dir = model_dir
        self.base_features = [
            'TIME OCC', 'AREA', 'Rpt Dist No', 'Part 1-2', 
            'Crm Cd', 'Vict Age', 'LAT', 'LON'
        ]
        self.setup_logging()
        
    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename=f'logs/enhanced_predictor_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def engineer_features(self, df):
        """Enhanced feature engineering"""
        df = df.copy()
        
        # Time-based features
        df['Hour'] = df['TIME OCC'] // 100
        df['Minute'] = df['TIME OCC'] % 100
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
        df['Is_Rush_Hour'] = ((df['Hour'].isin([7,8,9,16,17,18]))).astype(int)
        df['Time_Of_Day'] = pd.cut(df['Hour'], 
                                 bins=[0,6,12,18,24], 
                                 labels=['Night','Morning','Afternoon','Evening'])
        
        # Location features
        df['Dist_From_Center'] = np.sqrt(
            (df['LAT'] - df['LAT'].mean())**2 + 
            (df['LON'] - df['LON'].mean())**2
        )
        
        # Crime code features
        if 'Part 1-2' in df.columns:
            df['Is_Part_1'] = (df['Part 1-2'] == 1).astype(int)
        
        # Interaction features
        df['Time_Area_Interaction'] = df['Hour'] * df['AREA']
        df['Age_Time_Interaction'] = df['Vict Age'] * df['Hour']
        
        return df
    
    def train(self, df):
        try:
            self.logger.info("Starting model training")
            
            # Engineer features
            df_processed = self.engineer_features(df)
            
            # Prepare target
            le = LabelEncoder()
            if 'Crime_Group' in df.columns:
                df_processed['Target'] = le.fit_transform(df_processed['Crime_Group'])
            elif 'Target' not in df.columns:
                raise ValueError("No target variable found")
            
            # Save label encoder
            joblib.dump(le, os.path.join(self.model_dir, 'enhanced_label_encoder.joblib'))
            
            # Get all numeric features
            feature_cols = self.base_features + [
                'Hour', 'Minute', 'Is_Night', 'Is_Rush_Hour',
                'Dist_From_Center', 'Is_Part_1',
                'Time_Area_Interaction', 'Age_Time_Interaction'
            ]
            
            # Remove any features that don't exist in the dataframe
            feature_cols = [col for col in feature_cols if col in df_processed.columns]
            
            X = df_processed[feature_cols]
            y = df_processed['Target']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest with optimized parameters
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=3,
                min_samples_split=5,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
            
            print("\nTraining Random Forest...")
            rf_model.fit(X_scaled, y)
            
            # Save models and artifacts
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(rf_model, os.path.join(self.model_dir, 'enhanced_model.joblib'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'enhanced_scaler.joblib'))
            
            # Evaluate model
            y_pred = rf_model.predict(X_scaled)
            
            print("\nModel Performance:")
            report = classification_report(y, y_pred)
            print(report)
            
            # Save feature columns
            self.feature_cols = feature_cols
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            print(f"Training error: {str(e)}")
            return False
    
    def predict(self, input_data):
        try:
            # Load models
            model = joblib.load(os.path.join(self.model_dir, 'enhanced_model.joblib'))
            scaler = joblib.load(os.path.join(self.model_dir, 'enhanced_scaler.joblib'))
            le = joblib.load(os.path.join(self.model_dir, 'enhanced_label_encoder.joblib'))
            
            # Process input
            processed_data = self.engineer_features(input_data)
            X = processed_data[self.feature_cols]
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            # Get actual category name
            predicted_category = le.inverse_transform(prediction)[0]
            
            # Get feature importance
            importance = dict(zip(self.feature_cols, model.feature_importances_))
            top_features = dict(sorted(importance.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:5])
            
            return {
                'predicted_category': predicted_category,
                'confidence': float(max(probabilities[0])),
                'top_features': top_features
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            print(f"Prediction error: {str(e)}")
            raise