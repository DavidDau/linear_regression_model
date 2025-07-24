# Render Deployment Guide for Cardiovascular Disease Prediction API

This guide will walk you through deploying your FastAPI application to Render, a cloud platform for hosting web applications.

## Prerequisites

- GitHub account
- Render account (free tier available)
- Your project code in a GitHub repository

## Step 1: Prepare Your Project for Deployment

### 1.1 Update app.py for Production

Make sure your `app.py` file includes the following changes for production:

```python
import os
import uvicorn

# ... rest of your code ...

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )
```

### 1.2 Verify requirements.txt

Ensure your `requirements.txt` includes all necessary dependencies:

```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3
python-multipart==0.0.6
```

### 1.3 Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and create a new repository
2. Name it: `cardio-prediction-api`
3. Make it public (required for Render free tier)
4. Don't initialize with README (you already have one)

### 1.4 Push Your Code to GitHub

```bash
cd linear_regression_model
git init
git add .
git commit -m "Initial commit: Cardiovascular Disease Prediction API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cardio-prediction-api.git
git push -u origin main
```

## Step 2: Create a Render Account

1. Go to [render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up with your GitHub account (recommended)
4. Verify your email address

## Step 3: Deploy Your API to Render

### 3.1 Create a New Web Service

1. Once logged in, click "New +"
2. Select "Web Service"
3. Choose "Build and deploy from a Git repository"
4. Click "Next"

### 3.2 Connect Your GitHub Repository

1. If prompted, authorize Render to access your GitHub repositories
2. Find and select your `cardio-prediction-api` repository
3. Click "Connect"

### 3.3 Configure Your Web Service

Fill in the deployment configuration:

- **Name**: `cardio-prediction-api` (or any unique name)
- **Region**: Choose the closest to your target users
- **Branch**: `main`
- **Root Directory**: `summative/API` (since your API files are in this folder)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

### 3.4 Environment Settings

- **Plan**: Select "Free" (sufficient for testing)
- **Environment Variables**: None needed for basic setup

### 3.5 Deploy

1. Click "Create Web Service"
2. Render will start building and deploying your application
3. The process takes 5-10 minutes for the first deployment

## Step 4: Monitor Your Deployment

### 4.1 Check Build Logs

- In your Render dashboard, click on your service
- Go to the "Logs" tab to monitor the build process
- Look for successful installation of dependencies

### 4.2 Verify Deployment

Once deployed successfully:

- Your API will be available at: `https://your-service-name.onrender.com`
- Check the "Overview" tab for your service URL
- Visit the URL to see your API root response

## Step 5: Test Your Deployed API

### 5.1 Basic API Test

Visit these URLs in your browser:

- Root: `https://your-service-name.onrender.com/`
- Health: `https://your-service-name.onrender.com/health`
- Docs: `https://your-service-name.onrender.com/docs`

### 5.2 Test Prediction Endpoint

Use the interactive documentation at `/docs` to test the prediction endpoint:

1. Go to `https://your-service-name.onrender.com/docs`
2. Find the `POST /predict` endpoint
3. Click "Try it out"
4. Use this sample data:

```json
{
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
```

5. Click "Execute" and verify you get a prediction response

## Step 6: Update Your Flutter App

### 6.1 Update API Service

In your Flutter app, update `lib/services/api_service.dart`:

```dart
class ApiService {
  // Replace with your actual Render URL
  static const String baseUrl = 'https://your-service-name.onrender.com';

  // ... rest of your code ...
}
```

### 6.2 Test Flutter App

1. Run your Flutter app: `flutter run`
2. Try making a prediction to ensure it connects to your deployed API

## Step 7: Custom Domain (Optional)

### 7.1 Add Custom Domain

If you have a custom domain:

1. In Render dashboard, go to your service
2. Go to "Settings" tab
3. Scroll to "Custom Domains"
4. Click "Add Custom Domain"
5. Enter your domain and follow DNS configuration instructions

## Troubleshooting

### Common Issues and Solutions

#### 1. Build Fails - Dependencies Issue

**Problem**: Build fails during pip install

**Solution**:

- Check your `requirements.txt` for typos
- Ensure all package versions are compatible
- Try removing version numbers to use latest versions

#### 2. App Doesn't Start - Port Issue

**Problem**: App fails to start or shows port errors

**Solution**:
Ensure your `app.py` includes:

```python
import os
port = int(os.getenv("PORT", 8000))
```

#### 3. CORS Errors in Flutter App

**Problem**: Flutter app can't connect to API due to CORS

**Solution**:
Verify your FastAPI app includes CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 4. Model Files Not Found

**Problem**: API returns errors about missing model files

**Solution**:

- Ensure `prediction.py` runs on first API call to train and save models
- Check that model files are being created in the correct directory

#### 5. Cold Start Issues

**Problem**: API is slow to respond after inactivity

**Solution**:

- This is normal for Render's free tier (cold starts)
- Consider upgrading to a paid plan for production use
- The API will "wake up" after the first request

## Maintenance

### Updating Your Deployment

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push origin main
   ```
3. Render will automatically redeploy your application

### Monitoring

- Check the Render dashboard regularly for service health
- Review logs for any errors or issues
- Monitor API usage if needed

## Next Steps

After successful deployment:

1. Share your API URL with team members or in your project documentation
2. Update your Flutter app to use the production API
3. Consider setting up monitoring and analytics
4. Plan for scaling if you expect high traffic

## Support

If you encounter issues:

1. Check Render's [documentation](https://render.com/docs)
2. Review your service logs in the Render dashboard
3. Check GitHub repository for any deployment-related issues
4. Contact Render support for platform-specific problems

## Security Notes for Production

For production deployment, consider:

1. **Environment Variables**: Store sensitive configuration in Render environment variables
2. **Rate Limiting**: Implement API rate limiting
3. **Authentication**: Add API key authentication if needed
4. **HTTPS**: Render provides HTTPS by default
5. **Input Validation**: Ensure robust input validation (already implemented with Pydantic)

Your Cardiovascular Disease Prediction API is now live and accessible to users worldwide! 🎉
