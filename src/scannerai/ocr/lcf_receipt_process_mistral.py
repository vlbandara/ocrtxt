"""Use Mistral AI vision API to extract structured information from an image."""

import base64
import io
import json
import mimetypes
import os
import time

import cv2
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scannerai.utils.scanner_utils import merge_pdf_pages, read_api_key
from scannerai.utils.image_enhancer import optimize_image_size, enhance_image_for_ocr
from scannerai.utils.data_validator import validate_receipt_data, extract_json_from_text


class LCFReceiptProcessMistral:
    """OCR processor using Mistral AI vision API - Fast and reliable."""

    def __init__(self, mistral_api_key_path=None, mistral_api_key=None):
        """Initialize Mistral API with credentials."""
        self.InitSuccess = False
        self.api_key = None
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        # Try different pixtral models - they're fast and reliable for vision
        self.models_to_try = [
            "pixtral-12b-2409",      # Latest stable pixtral
            "pixtral-large-latest",  # Latest pixtral (auto-updates)
            "pixtral-small-latest",   # Smaller, faster pixtral
        ]
        self.model = None  # Will be set after testing
        
        # Accept either API key path, direct API key, or environment variable
        if mistral_api_key:
            self.api_key = mistral_api_key
        elif os.getenv("MISTRAL_API_KEY"):
            self.api_key = os.getenv("MISTRAL_API_KEY")
        elif mistral_api_key_path and os.path.exists(mistral_api_key_path):
            self.api_key = read_api_key(mistral_api_key_path, env_var="MISTRAL_API_KEY")
        else:
            print("WARNING: Mistral API key not provided! Check MISTRAL_API_KEY env var or provide mistral_api_key_path.")
            return

        # Prompt for structured receipt extraction
        self.prompt = """Analyze this receipt image and extract the following information in JSON format:
{
    "shop_name": "store name",
    "items": [
        {"name": "item name", "price": 1.99},
        {"name": "item name", "price": 2.49}
    ],
    "total_amount": 27.83,
    "payment_mode": "card or cash"
}

Important:
- Extract all items with their names and prices
- If any value is missing, use null
- Do not leave item names empty
- Return only valid JSON, no markdown formatting
"""

        # Test which model works
        self.model = self._find_working_model()
        
        # Create session with connection pooling and retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("https://", adapter)
        
        if self.model:
            self.InitSuccess = True
            print(f"✅ Mistral AI OCR processor initialized successfully with model: {self.model}")
        else:
            print("⚠️ Warning: Could not verify Mistral model availability. Will try during first request.")
            self.InitSuccess = True  # Still allow initialization, will fail gracefully on first use
            self.model = self.models_to_try[0]  # Use first model as default

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def _find_working_model(self):
        """Test which model is available (quick test without image)."""
        # Just return the first model - actual availability will be tested on first request
        # This avoids unnecessary API calls during initialization
        return self.models_to_try[0] if self.models_to_try else None

    @staticmethod
    def encode_image(image):
        """Encode an image to base64 for API."""
        buffered = io.BytesIO()
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def process_receipt(
        self, file_path, debug_mode=False, enable_price_count=False, enhance_image=True
    ):
        """Extract structured information from an input image - Fast and reliable."""
        
        start_time = time.time()
        file_type, _ = mimetypes.guess_type(file_path)

        # Load and optimize image
        if file_type == "application/pdf":
            image = merge_pdf_pages(file_path)
        elif file_type and file_type.startswith("image/"):
            image = Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if debug_mode:
            opencv_img = np.array(image)
            opencv_img = opencv_img[:, :, ::-1].copy()
            cv2.imshow(f"input image: {file_path}", opencv_img)
            cv2.waitKey(0)

        # Enhanced image preprocessing
        original_size = image.size
        if enhance_image:
            # Optimize size first
            image = optimize_image_size(image, max_dimension=2048, min_dimension=512)
            # Enhance for better OCR
            image = enhance_image_for_ocr(image, enhance_contrast=True, denoise=True, sharpen=False)
            if enable_price_count and image.size != original_size:
                print(f"📐 Optimized image from {original_size} to {image.size}")
        else:
            # Just optimize size without enhancement
            image = optimize_image_size(image, max_dimension=2048, min_dimension=512)

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "total_amount": None,
            "items": [],
            "receipt_pathfile": file_path,
        }

        if not self.InitSuccess or not self.api_key:
            return receipt_data

        # Encode image
        base64_image = self.encode_image(image)

        # Prepare API request with retry logic
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                # Try different models if one fails
                models_to_attempt = [self.model] + [m for m in self.models_to_try if m != self.model]
                response = None
                request_successful = False
                
                for model_name in models_to_attempt:
                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.1,  # Low temperature for consistent output
                        "max_tokens": 2000,
                    }

                    # Make API request with timeout using session (connection pooling)
                    response = self.session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=60  # 60 second timeout
                    )

                    if response.status_code == 200:
                        # Update working model
                        if self.model != model_name:
                            print(f"✅ Using model: {model_name}")
                            self.model = model_name
                        request_successful = True
                        break
                    elif response.status_code == 404:
                        # Model not found, try next one
                        continue
                    else:
                        # Other error, break and handle
                        break
                
                if request_successful and response and response.status_code == 200:
                    result = response.json()
                    receipt_info = result["choices"][0]["message"]["content"]
                    
                    # Enhanced JSON extraction
                    json_str = extract_json_from_text(receipt_info)
                    if not json_str:
                        print("⚠️ No JSON found in response")
                        return receipt_data

                    try:
                        receipt_data = json.loads(json_str)
                        receipt_data["receipt_pathfile"] = file_path
                        
                        # Validate and clean extracted data
                        receipt_data = validate_receipt_data(receipt_data)
                        
                        # Performance tracking
                        processing_time = time.time() - start_time
                        if enable_price_count:
                            usage = result.get("usage", {})
                            print(f"⚡ Processing time: {processing_time:.2f}s | "
                                  f"Tokens: {usage.get('total_tokens', 0)} "
                                  f"(Input: {usage.get('prompt_tokens', 0)}, "
                                  f"Output: {usage.get('completion_tokens', 0)})")
                        
                        return receipt_data
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse JSON: {str(e)[:200]}")
                        print(f"Response (first 500 chars): {json_str[:500] if json_str else receipt_info[:500]}")
                        return receipt_data
                        
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⚠️ Rate limit hit (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ Rate limit exceeded after all retries")
                        return receipt_data
                else:
                    error_msg = f"API error {response.status_code}: {response.text[:200]}"
                    print(f"❌ {error_msg}")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return receipt_data

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                error_type = "Timeout" if isinstance(e, requests.exceptions.Timeout) else "Connection"
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️ {error_type} error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Request {error_type.lower()} after all retries")
                    return receipt_data
                    
            except Exception as e:
                error_str = str(e)
                print(f"❌ Error processing receipt: {error_str[:200]}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return receipt_data

        return receipt_data

