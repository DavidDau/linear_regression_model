# Cardiovascular Disease Prediction System

A comprehensive machine learning system for predicting cardiovascular disease risk, featuring data analysis, API backend, and Flutter mobile application.

## My Mission

To mitigate labor shortages and boost productivity by leveraging predictive analytics to identify and prevent cardio-related diseases early, ensuring a healthier workforce and a more resilient economy.

## Project Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── cardio_base.csv         # Dataset
│   │   └── multivariate.ipynb      # Analysis notebook
│   ├── API/
│   │   ├── prediction.py           # Model training
│   │   ├── app.py                  # FastAPI application
│   │   ├── requirements.txt        # Python dependencies
│   │   └── test_predictions.py     # API tests
│   └── FlutterApp/                 # Mobile application
│       ├── lib/
│       └── pubspec.yaml
├── README.md
└── .gitignore
```

## Features

### Machine Learning Models

- Linear Regression with Gradient Descent
- Decision Tree Classifier
- Random Forest Classifier
- Model comparison and evaluation
- Feature engineering and standardization

### FastAPI Backend

- RESTful API endpoints
- Data validation with Pydantic
- CORS support for cross-origin requests
- Interactive Swagger UI documentation
- Health check and model info endpoints

### Flutter Mobile App

- Multi-page interface
- Form validation and error handling
- Risk level visualization
- Beautiful, organized UI design

## Local Development

### Backend Setup

```bash
cd summative/API
pip install -r requirements.txt
python prediction.py  # Train models
python app.py         # Start API server
```

### Flutter Setup

```bash
cd summative/FlutterApp
flutter pub get
flutter run
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model details
- `POST /predict` - Make predictions
- `GET /docs` - Interactive Swagger UI

## Mobile App Features

### Input Fields (11 parameters)

- Age, Gender, Height, Weight
- Systolic/Diastolic Blood Pressure
- Cholesterol, Glucose levels
- Smoking, Alcohol consumption, Physical Activity

### Risk Assessment

- **Low Risk** (0-33%): Green indicator
- **Moderate Risk** (34-66%): Orange indicator
- **High Risk** (67-100%): Red indicator

## Model Performance

All models achieved perfect performance (R² = 1.0) on the synthetic target:

- Linear Regression: R² = 1.0000
- Decision Tree: R² = 1.0000
- Random Forest: R² = 1.0000

## Requirements

- Python 3.8+
- Flutter 3.0+
- FastAPI, scikit-learn, pandas, numpy
- HTTP package for Flutter

## Deployment

This project is configured for deployment on Render.com with automatic GitHub integration.

## Disclaimer

This system is for educational and research purposes. Always consult healthcare professionals for medical advice.

## License

This project is part of an academic assignment for machine learning coursework.
