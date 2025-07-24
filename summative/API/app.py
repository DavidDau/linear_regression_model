"""
FastAPI Application for Cardiovascular Disease Prediction
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np
from prediction import predict_cardiovascular_disease

# Initialize FastAPI app
app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    description="API for predicting cardiovascular disease risk using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years (1-120)")
    gender: int = Field(..., ge=1, le=2, description="Gender (1=female, 2=male)")
    height: int = Field(..., ge=100, le=250, description="Height in cm (100-250)")
    weight: int = Field(..., ge=30, le=300, description="Weight in kg (30-300)")
    ap_hi: int = Field(..., ge=70, le=250, description="Systolic blood pressure (70-250)")
    ap_lo: int = Field(..., ge=40, le=150, description="Diastolic blood pressure (40-150)")
    cholesterol: int = Field(..., ge=1, le=3, description="Cholesterol level (1=normal, 2=above normal, 3=well above normal)")
    gluc: int = Field(..., ge=1, le=3, description="Glucose level (1=normal, 2=above normal, 3=well above normal)")
    smoke: int = Field(..., ge=0, le=1, description="Smoking status (0=no, 1=yes)")
    alco: int = Field(..., ge=0, le=1, description="Alcohol consumption (0=no, 1=yes)")
    active: int = Field(..., ge=0, le=1, description="Physical activity (0=no, 1=yes)")

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "gender": 1,
                "height": 170,
                "weight": 75,
                "ap_hi": 130,
                "ap_lo": 85,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }

# Response model
class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Probability of cardiovascular disease (0-1)")
    risk_level: str = Field(..., description="Risk level (Low, Moderate, High)")
    has_disease: bool = Field(..., description="Predicted disease status (True/False)")
    message: str = Field(..., description="Interpretation message")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cardiovascular Disease Prediction API",
        "version": "1.0.0",
        "description": "API for predicting cardiovascular disease risk",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running successfully"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(patient_data: PatientData) -> PredictionResponse:
    """
    Predict cardiovascular disease risk for a patient
    
    Args:
        patient_data: Patient information including age, gender, vital signs, etc.
        
    Returns:
        Prediction result with probability, risk level, and interpretation
    """
    try:
        # Convert Pydantic model to dictionary
        input_data = patient_data.dict()
        
        # Make prediction
        result = predict_cardiovascular_disease(input_data)
        
        # Create interpretation message
        prob_percent = result["probability"] * 100
        
        if result["risk_level"] == "Low":
            message = f"Low risk ({prob_percent:.1f}% probability). Continue healthy lifestyle practices."
        elif result["risk_level"] == "Moderate":
            message = f"Moderate risk ({prob_percent:.1f}% probability). Consider lifestyle modifications and regular check-ups."
        else:
            message = f"High risk ({prob_percent:.1f}% probability). Recommend immediate medical consultation and intervention."
        
        return PredictionResponse(
            probability=result["probability"],
            risk_level=result["risk_level"],
            has_disease=result["has_disease"],
            message=message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the machine learning model"""
    return {
        "model_type": "Ensemble (Random Forest/Linear Regression/Decision Tree)",
        "features": [
            "age", "gender", "ap_hi", "ap_lo", "cholesterol", 
            "gluc", "smoke", "alco", "active", "bmi", 
            "pulse_pressure", "age_risk", "bp_risk", "lifestyle_risk"
        ],
        "feature_descriptions": {
            "age": "Age in years",
            "gender": "Gender (1=female, 2=male)",
            "ap_hi": "Systolic blood pressure",
            "ap_lo": "Diastolic blood pressure",
            "cholesterol": "Cholesterol level (1-3)",
            "gluc": "Glucose level (1-3)",
            "smoke": "Smoking status (0/1)",
            "alco": "Alcohol consumption (0/1)",
            "active": "Physical activity (0/1)",
            "bmi": "Body Mass Index (calculated)",
            "pulse_pressure": "Pulse pressure (calculated)",
            "age_risk": "Age risk factor (calculated)",
            "bp_risk": "Blood pressure risk (calculated)",
            "lifestyle_risk": "Lifestyle risk score (calculated)"
        },
        "target": "Cardiovascular disease probability (0-1)"
    }

@app.on_event("startup")
def startup_event():
    """Load the machine learning model at startup"""
    try:
        load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
