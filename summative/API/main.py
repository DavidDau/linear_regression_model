from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import API_URL

app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    description="API for predicting cardiovascular disease risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Cardiovascular Disease Prediction API",
        "docs": f"{API_URL}/docs",
        "redoc": f"{API_URL}/redoc"
    }