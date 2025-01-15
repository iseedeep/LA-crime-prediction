# LA Crime Prediction Model

Advanced machine learning model for predicting crime categories in Los Angeles using LAPD data.

## Overview
- Predicts crime categories using ensemble machine learning
- Uses LAPD historical crime data (2020-2023)
- Features include location, time, demographic data, and derived features
- Includes confidence scoring and feature importance analysis
- Supports both basic and enhanced prediction models

## Crime Categories
Current model handles four main crime categories:
- PROPERTY_CRIME (62.8%)
- VIOLENT_CRIME (24.3%)
- MISCELLANEOUS (7.7%)
- OTHER_CRIME (5.2%)

## Project Structure
```
safety_prediction_project/
├── src/
│   ├── api/                # FastAPI server
│   ├── data_pipeline/      # Data processing
│   ├── models/
│   │   ├── crime_predictor.py      # Base predictor
│   │   ├── enhanced_predictor.py   # Enhanced model
│   │   └── train_model.py          # Training
│   └── utils/              # Utilities
├── enhanced_models/        # Enhanced model files
├── trained_models/         # Base model files
├── data.csv               # Training data
└── requirements.txt
```

## Features
### Base Features
- Time: 24-hour format (0-2400)
- Area: LAPD division number (1-21)
- District: Reporting district number
- Crime Code: LAPD crime classification
- Location: LA coordinates (latitude/longitude)

### Enhanced Features
- Temporal patterns (time of day, rush hours)
- Location clustering
- Crime hotspot analysis
- Demographic interactions
- Feature importance ranking

## Installation
```bash
# Create environment
conda create -n crime_pred python=3.8
conda activate crime_pred

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Basic Prediction
```python
from src.models.crime_predictor import CrimePredictor
import pandas as pd

# Initialize basic predictor
predictor = CrimePredictor(model_dir='trained_models')

# Example: Downtown LA prediction
input_data = pd.DataFrame({
    'TIME OCC': [1200],      # 12:00 PM
    'AREA': [1],             # Central Division
    'Rpt Dist No': [123],    # Reporting District
    'Part 1-2': [1],         # Crime Category
    'Crm Cd': [624],         # Crime Code
    'Vict Age': [30],        # Victim Age
    'LAT': [34.0522],        # Downtown LA latitude
    'LON': [-118.2437]       # Downtown LA longitude
})

result = predictor.predict(input_data)
print(f"Predicted category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Enhanced Prediction
```python
from src.models.enhanced_predictor import EnhancedCrimePredictor

# Initialize enhanced predictor
predictor = EnhancedCrimePredictor(model_dir='enhanced_models')

# Make prediction with the same input data
result = predictor.predict(input_data)
print(f"Predicted category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nTop contributing features:")
for feature, importance in result['top_features'].items():
    print(f"- {feature}: {importance:.4f}")
```

## API Service
The model is also available through a REST API:
```bash
# Start the API server
uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000
```

Example API request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "time_occ": 1200,
         "area": 1,
         "rpt_dist_no": 123,
         "part_1_2": 1,
         "crm_cd": 624,
         "vict_age": 30,
         "lat": 34.0522,
         "lon": -118.2437
     }'
```

## Model Performance
The enhanced model provides:
- Higher prediction confidence
- Feature importance analysis
- Detailed performance metrics
- More sophisticated feature engineering

## Future Enhancements
- Real-time prediction updates
- Spatial-temporal pattern analysis
- Integration with additional data sources
- Advanced visualization tools

## Data Source
Uses LAPD Crime Data from 2020 to Present (data.lacity.org)
