import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import os
import joblib

def group_crime_types(crime_desc):
    violent_crimes = ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'KIDNAP']
    property_crimes = ['BURGLARY', 'THEFT', 'STOLEN', 'VANDALISM']
    other_crimes = ['TRESPASSING', 'WEAPON', 'DRUGS']
    
    crime_upper = crime_desc.upper()
    if any(crime in crime_upper for crime in violent_crimes):
        return 'VIOLENT_CRIME'
    elif any(crime in crime_upper for crime in property_crimes):
        return 'PROPERTY_CRIME'
    elif any(crime in crime_upper for crime in other_crimes):
        return 'OTHER_CRIME'
    return 'MISCELLANEOUS'

def train_and_evaluate(df, output_dir, sample_size=50000):
    try:
        # Group crime types
        df['Crime_Group'] = df['Crm Cd Desc'].apply(group_crime_types)
        
        # Create and encode target
        le = LabelEncoder()
        df['Target'] = le.fit_transform(df['Crime_Group'])
        
        # Select features
        features = [
            'TIME OCC', 'AREA', 'Rpt Dist No', 'Part 1-2', 
            'Crm Cd', 'Vict Age', 'LAT', 'LON'
        ]
        
        # Sample data
        df_sampled = df.sample(min(len(df), sample_size), random_state=42)
        
        # Prepare features and target
        X = df_sampled[features]
        y = df_sampled['Target']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train model
        print("\nTraining balanced model...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_balanced, y_train_balanced)
        
        # Save artifacts
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, 'crime_prediction_model.joblib'))
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))
        
        # Evaluate
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    local_path = "data.csv"
    output_dir = 'trained_models'
    df = pd.read_csv(local_path)
    success = train_and_evaluate(df, output_dir)