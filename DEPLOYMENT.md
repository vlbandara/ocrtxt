# Deployment Guide for Streamlit Cloud

This guide will help you deploy the Receipt Scanner application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. A Streamlit Cloud account (free at https://streamlit.io/cloud)
3. API keys for your chosen OCR service:
   - Gemini API key (if using OCR_MODEL=3)
   - OpenAI API key (if using OCR_MODEL=1 or 2)
   - Mistral API key (if using OCR_MODEL=4)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `receipt-scanner`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)

## Step 2: Push Code to GitHub

Run these commands in your terminal:

```bash
cd /Users/vinodlahiru/Downloads/receipt_scanner-main
git init
git add .
git commit -m "Initial commit: Receipt Scanner app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/receipt-scanner.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select your repository: `YOUR_USERNAME/receipt-scanner`
5. Set the main file path: `streamlit_app.py`
6. Click "Advanced settings" and configure:

### Environment Variables (Secrets)

Click "Secrets" and add the following based on your OCR model choice:

**For Mistral AI (OCR_MODEL=4 - Recommended):**
```
MISTRAL_API_KEY=your_mistral_api_key_here
OCR_MODEL=4
CLASSIFIER_MODEL_PATH=src/scannerai/classifiers/trainedModels/LRCountVectorizer.sav
LABEL_ENCODER_PATH=src/scannerai/classifiers/trainedModels/encoder.pkl
```

**For Gemini (OCR_MODEL=3):**
```
GEMINI_API_KEY=your_gemini_api_key_here
OCR_MODEL=3
CLASSIFIER_MODEL_PATH=src/scannerai/classifiers/trainedModels/LRCountVectorizer.sav
LABEL_ENCODER_PATH=src/scannerai/classifiers/trainedModels/encoder.pkl
```

**For OpenAI (OCR_MODEL=1 or 2):**
```
OPENAI_API_KEY=your_openai_api_key_here
OCR_MODEL=1
CLASSIFIER_MODEL_PATH=src/scannerai/classifiers/trainedModels/LRCountVectorizer.sav
LABEL_ENCODER_PATH=src/scannerai/classifiers/trainedModels/encoder.pkl
```

### Optional Configuration Variables

```
DEBUG_MODE=False
ENABLE_PREPROCESSING=False
SAVE_PROCESSED_IMAGE=False
ENABLE_PRICE_COUNT=True
```

7. Click "Deploy!"

## Step 4: Verify Deployment

Once deployed, Streamlit Cloud will provide you with a URL like:
`https://YOUR_APP_NAME.streamlit.app`

Visit the URL and test the application by uploading a receipt image.

## Troubleshooting

### ImportError with OpenCV (cv2)
If you see an error like `ImportError: This app has encountered an error` related to `cv2`:
- The app uses `opencv-python-headless` which is better for server environments
- Make sure you've pulled the latest code with the updated `requirements.txt`
- Redeploy the app after updating requirements.txt

### App fails to start
- Check that all required environment variables are set in Streamlit Cloud secrets
- Verify that the API keys are correct
- Check the logs in Streamlit Cloud dashboard
- Ensure Python version is 3.11 or 3.12 (set in Streamlit Cloud app settings if needed)

### OCR not working
- Verify your API key is valid and has sufficient credits/quota
- Check that the OCR_MODEL environment variable matches your API key type

### Model files not found
- Ensure the classifier model files are committed to the repository
- Check that CLASSIFIER_MODEL_PATH and LABEL_ENCODER_PATH point to the correct relative paths

## Notes

- The app will automatically use environment variables for API keys when deployed on Streamlit Cloud
- Model files must be committed to the repository (they're already included)
- The app supports multiple OCR providers - choose based on your preference and API availability

