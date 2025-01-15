from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import Dict, Any

class PredictionInput(BaseModel):
    time_occ: int
    area: int
    rpt_dist_no: int
    part_1_2: int
    crm_cd: int
    vict_age: int
    lat: float
    lon: float

class Predictor:
    def __init__(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, 'crime_prediction_model.joblib'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))

    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        scaled_features = self.scaler.transform(input_data)
        prediction = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)
        
        return {
            'predicted_category': self.label_encoder.inverse_transform(prediction),
            'confidence': [max(prob) for prob in probabilities]
        }

    def explain_prediction(self, input_data: pd.DataFrame, prediction: Dict[str, Any]) -> Dict[str, str]:
        return {
            'interpretation': f"Predicted crime category: {prediction['predicted_category'][0]} "
                            f"with {prediction['confidence'][0]:.2%} confidence"
        }

app = FastAPI(title="Crime Prediction API")
predictor = Predictor(model_dir="trained_models")

@app.post("/predict")
async def predict_crime(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = predictor.predict(input_df)
        
        # Get explanation
        explanation = predictor.explain_prediction(input_df, prediction)
        
        return {
            'status': 'success',
            'prediction': prediction['predicted_category'][0],
            'confidence': float(prediction['confidence'][0]),
            'explanation': explanation['interpretation']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {'status': 'healthy'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)