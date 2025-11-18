"""use Gemini to process an image and output structured information."""

import json
import mimetypes
import os
import time

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from scannerai.utils.scanner_utils import merge_pdf_pages, read_api_key


class LCFReceiptProcessGemini:
    """OCR processor using Gemini."""

    def __init__(self, google_credentials_path=None, gemini_api_key_path=None, gemini_api_key=None):
        """Initialize Gemini API with credentials.
        
        Args:
            google_credentials_path: Path to Google Cloud credentials file (optional)
            gemini_api_key_path: Path to Gemini API key file (optional)
            gemini_api_key: Direct Gemini API key string (optional, takes precedence)
        """

        self.model = None
        self.InitSuccess = False

        # Try to get API key - prioritize direct key, then env var, then file
        gemini_api_key_value = None
        if gemini_api_key:
            gemini_api_key_value = gemini_api_key
        elif os.getenv("GEMINI_API_KEY"):
            gemini_api_key_value = os.getenv("GEMINI_API_KEY")
        elif gemini_api_key_path:
            gemini_api_key_value = read_api_key(gemini_api_key_path, env_var="GEMINI_API_KEY")
        
        if not gemini_api_key_value:
            print("WARNING: Gemini API key not found! Check GEMINI_API_KEY env var or provide gemini_api_key_path.")
            return

        # IMPORTANT: For free Gemini API, we should NOT use Google Cloud credentials
        # Google Cloud credentials are only for Vertex AI (paid/enterprise)
        # Clear any existing credentials to ensure we use the free API
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            print("Note: Clearing GOOGLE_APPLICATION_CREDENTIALS to use free Gemini API")
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
        # Only set credentials if explicitly provided AND we want Vertex AI
        # For now, we'll use the free API, so we skip setting credentials
        # if google_credentials_path and os.path.exists(google_credentials_path):
        #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path

        genai.configure(api_key=gemini_api_key_value)

        # List available models and find one that supports vision
        print("Searching for available Gemini models...")
        available_models = []
        available_model_names = []  # Store both full names and short names
        try:
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    # Keep both full name and short name
                    full_name = model.name  # e.g., "models/gemini-pro-vision"
                    model_name = model.name.replace('models/', '')  # e.g., "gemini-pro-vision"
                    # Skip experimental models (they often have quota issues)
                    if '-exp' not in model_name.lower() and 'experimental' not in model_name.lower():
                        available_models.append((full_name, model_name))
                        available_model_names.append(model_name)
                        print(f"  Found: {model_name} (full: {full_name})")
                    else:
                        print(f"  Skipping experimental model: {model_name}")
        except Exception as e:
            print(f"Could not list models: {e}")
            available_models = []
            available_model_names = []
        
        # Preferred models in order of preference (only if they're actually available)
        # Prioritize stable (non-preview) Flash models for speed and reliability
        # Flash models are faster and have better quotas, ideal for receipt processing
        preferred_models = [
            "gemini-2.5-flash",           # Latest stable flash model - fast, good quota (BEST)
            "gemini-2.0-flash",           # Stable flash model - fast
            "gemini-2.0-flash-001",        # Stable flash variant - fast
            "gemini-flash-latest",        # Latest flash (auto-updates) - fast
            "gemini-1.5-flash",           # Older stable flash option - fast
            "gemini-2.5-pro",             # Latest stable pro model - slower but more capable
            "gemini-pro-latest",          # Latest pro (auto-updates) - slower
            "gemini-pro-vision",          # Legacy stable model
            "gemini-1.5-pro",             # Older pro model
            "gemini-pro"                  # Legacy text model
        ]
        
        # Build list of models to try: prioritize available models that match preferred
        models_to_try = []
        
        # Helper function to check if a model is a preview/experimental version
        def is_preview_model(name):
            return 'preview' in name.lower() or 'exp' in name.lower() or 'experimental' in name.lower()
        
        # First, add preferred stable (non-preview) models that are actually available
        for preferred in preferred_models:
            # Check if this preferred model is in the available list
            for full_name, short_name in available_models:
                if short_name == preferred and not is_preview_model(short_name):
                    models_to_try.append((full_name, short_name))
                    break
        
        # Then add preferred preview models (as fallback if no stable versions found)
        for preferred in preferred_models:
            # Check if this preferred model is in the available list (including previews)
            for full_name, short_name in available_models:
                if short_name == preferred and is_preview_model(short_name):
                    if (full_name, short_name) not in models_to_try:
                        models_to_try.append((full_name, short_name))
                    break
        
        # Finally, add any other available stable models that weren't in preferred list
        for full_name, short_name in available_models:
            if (full_name, short_name) not in models_to_try and not is_preview_model(short_name):
                models_to_try.append((full_name, short_name))
        
        # Last resort: add any remaining preview models
        for full_name, short_name in available_models:
            if (full_name, short_name) not in models_to_try:
                models_to_try.append((full_name, short_name))
        
        # If no models were found from API, try preferred models as fallback
        if not models_to_try:
            print("Warning: Could not list models from API, trying preferred models as fallback...")
            for preferred in preferred_models:
                models_to_try.append((f"models/{preferred}", preferred))
        
        # Try each model
        for full_name, model_name in models_to_try:
            try:
                # Try with full name first, then short name
                try:
                    self.model = genai.GenerativeModel(full_name)
                except:
                    self.model = genai.GenerativeModel(model_name)
                # Test if it actually works by checking if it's initialized
                print(f"✅ Successfully initialized model: {model_name}")
                break
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"❌ Failed to initialize {model_name}: {error_msg}")
                continue
        
        if self.model is None:
            print("\n❌ Could not initialize any Gemini model.")
            if available_model_names:
                print("Available models were:", ", ".join(available_model_names))
            else:
                print("Could not list available models from API")
            raise ValueError("Could not initialize any Gemini model. Please check your API key and model availability.")
        self.prompt = """Analyze this receipt image and extract the shop name, items with their prices, total amount, and payment mode. Format the output as a JSON object with the following structure:
        {
            "shop_name": "example shop",
            "items": [
                {"name": "item1", "price": 1.99},
                {"name": "item2", "price": 2.49},
                ...
            ],
            "total_amount": 27.83,
            "payment_mode": "card"
        }
        """

        self.InitSuccess = True

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def process_receipt(
        self, file_path, debug_mode=False, enable_price_count=False
    ):  # Add parameters for flexibility
        """Extract structured information from an input image."""

        file_type, _ = mimetypes.guess_type(file_path)

        if file_type == "application/pdf":
            image = merge_pdf_pages(file_path)
        elif file_type and file_type.startswith("image/"):
            image = Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if debug_mode:  # Use the passed debug_mode parameter
            opencv_img = np.array(image)
            opencv_img = opencv_img[:, :, ::-1].copy()
            cv2.imshow(f"input image: {file_path}", opencv_img)
            cv2.waitKey(0)

        if enable_price_count:
            print("input image size: ", image.size)

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "total_amount": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": file_path,
        }

        if self.model:
            # Optimize image size if it's too large (reduce processing time and timeout risk)
            max_dimension = 2048  # Maximum width or height
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                if enable_price_count:
                    print(f"Resized image from {image.size} to {new_size}")
            
            # Retry logic with exponential backoff for timeout errors
            max_retries = 3
            retry_delay = 2  # seconds
            response = None
            
            for attempt in range(max_retries):
                try:
                    # Configure generation with timeout and retry settings
                    generation_config = {
                        "temperature": 0.1,  # Lower temperature for more consistent output
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                    
                    # Make the API call with increased timeout
                    response = self.model.generate_content(
                        [self.prompt, image],
                        generation_config=generation_config,
                        request_options={"timeout": 120}  # 120 seconds timeout
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    is_timeout = "DeadlineExceeded" in error_str or "504" in error_str or "timeout" in error_str.lower()
                    
                    if attempt < max_retries - 1 and is_timeout:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Timeout error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # If it's the last attempt or not a timeout error, raise it
                        print(f"❌ Error processing receipt: {error_str[:200]}")
                        if not is_timeout:
                            raise  # Re-raise non-timeout errors immediately
                        # For timeout on last attempt, return empty receipt_data
                        return receipt_data

            if response is None:
                print("❌ Failed to get response from Gemini API after all retries")
                return receipt_data

            if enable_price_count:
                print("token usage:\n", response.usage_metadata)

            receipt_info = response.text
            lpos = receipt_info.find("{")
            receipt_info = receipt_info[lpos:]
            rpos = receipt_info.rfind("}")
            receipt_info = receipt_info[: rpos + 1]

            try:
                receipt_data = json.loads(receipt_info)
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON response: {str(e)[:200]}")
                print(f"Response text (first 500 chars): {receipt_info[:500]}")
                return receipt_data  # Return empty receipt_data on parse error

        receipt_data["receipt_pathfile"] = file_path

        return receipt_data
