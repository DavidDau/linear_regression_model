"""
Test script for the Cardiovascular Disease Prediction API
"""

import requests
import json

# API base URL (change this when deployed to Render)
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the API endpoints"""
    
    print("🧪 Testing Cardiovascular Disease Prediction API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    # Test model info endpoint
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    # Test prediction endpoint with sample data
    print("\n4. Testing prediction endpoint...")
    
    # Test cases
    test_cases = [
        {
            "name": "Low Risk Patient",
            "data": {
                "age": 30,
                "gender": 1,
                "height": 165,
                "weight": 60,
                "ap_hi": 110,
                "ap_lo": 70,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        },
        {
            "name": "Moderate Risk Patient",
            "data": {
                "age": 45,
                "gender": 2,
                "height": 175,
                "weight": 80,
                "ap_hi": 130,
                "ap_lo": 85,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 1,
                "alco": 0,
                "active": 1
            }
        },
        {
            "name": "High Risk Patient",
            "data": {
                "age": 60,
                "gender": 2,
                "height": 170,
                "weight": 95,
                "ap_hi": 160,
                "ap_lo": 100,
                "cholesterol": 3,
                "gluc": 3,
                "smoke": 1,
                "alco": 1,
                "active": 0
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n4.{i} Testing {test_case['name']}:")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {json.dumps(result, indent=2)}")
            else:
                print(f"Error Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
    
    # Test invalid data
    print("\n5. Testing invalid data handling...")
    invalid_data = {
        "age": 250,  # Invalid age
        "gender": 3,  # Invalid gender
        "height": 50,  # Invalid height
        "weight": 10,  # Invalid weight
        "ap_hi": 300,  # Invalid blood pressure
        "ap_lo": 200,  # Invalid blood pressure
        "cholesterol": 5,  # Invalid cholesterol
        "gluc": 5,  # Invalid glucose
        "smoke": 2,  # Invalid smoke
        "alco": 2,  # Invalid alcohol
        "active": 2  # Invalid active
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API testing completed!")
    print("\n📋 Instructions:")
    print("1. Make sure the API server is running: python app.py")
    print("2. API will be available at: http://localhost:8000")
    print("3. Interactive docs at: http://localhost:8000/docs")
    print("4. ReDoc documentation at: http://localhost:8000/redoc")

if __name__ == "__main__":
    test_api()
