"""Class to hold config parameters."""

import os
from io import StringIO

from dotenv import load_dotenv


def load_config(config_file=None):
    """Load configuration from a text file or environment variables.
    
    For Streamlit Cloud, configuration can be loaded from environment variables
    or from a config file. Environment variables take precedence.
    """
    config = {}
    
    # Try to load from config file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config_content = f.read()
            # Use StringIO to create a file-like object from the config content
            config_stream = StringIO(config_content)
            load_dotenv(stream=config_stream)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found: {config_file}")
    elif config_file:
        print(f"Warning: Configuration file not found: {config_file}")
    
    # Also load from .env file if it exists (for local development)
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    
    # Load configuration - environment variables take precedence
    config = {
        # Debug and Processing Settings
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "ENABLE_PREPROCESSING": os.getenv(
            "ENABLE_PREPROCESSING", "False"
        ).lower()
        == "true",
        "SAVE_PROCESSED_IMAGE": os.getenv(
            "SAVE_PROCESSED_IMAGE", "False"
        ).lower()
        == "true",
        "ENABLE_PRICE_COUNT": os.getenv("ENABLE_PRICE_COUNT", "False").lower()
        == "true",
        # OCR Model Settings
        "OCR_MODEL": int(os.getenv("OCR_MODEL", "3")),  # Default to Gemini
        # Classification Model Paths
        "CLASSIFIER_MODEL_PATH": os.getenv("CLASSIFIER_MODEL_PATH"),
        "LABEL_ENCODER_PATH": os.getenv("LABEL_ENCODER_PATH"),
        # API Keys - support both file paths and direct keys from env vars
        "GEMINI_API_KEY_PATH": os.getenv("GEMINI_API_KEY_PATH"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),  # Direct key from env
        "OPENAI_API_KEY_PATH": os.getenv("OPENAI_API_KEY_PATH"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # Direct key from env
        "MISTRAL_API_KEY_PATH": os.getenv("MISTRAL_API_KEY_PATH"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),  # Direct key from env
        # Google Cloud credentials
        "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH"),
    }

    # Validate required configurations
    # required_configs = ['CLASSIFIER_MODEL_PATH', 'LABEL_ENCODER_PATH']
    # missing_configs = [key for key in required_configs if not config[key]]

    # if missing_configs:
    #     raise ValueError(f"Missing required configuration(s): {', '.join(missing_configs)}")

    return config


# Create a Config class to handle configuration
class Config:
    """Class to handle configuration."""

    _instance = None
    _config = None

    def __new__(cls, config_file):
        """Create a configuration instance with config_file."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = load_config(config_file)
        return cls._instance

    @property
    def debug_mode(self):
        """Get DEBUG_MODE."""
        return self._config["DEBUG_MODE"]

    @property
    def enable_preprocessing(self):
        """Get ENABLE_PREPROCESSING."""
        return self._config["ENABLE_PREPROCESSING"]

    @property
    def save_processed_image(self):
        """Get SAVE_PROCESSED_IMAGE."""
        return self._config["SAVE_PROCESSED_IMAGE"]

    @property
    def enable_price_count(self):
        """Get ENABLE_PRICE_COUNT."""
        return self._config["ENABLE_PRICE_COUNT"]

    @property
    def ocr_model(self):
        """Get OCR_MODEL."""
        return self._config["OCR_MODEL"]

    @property
    def classifier_model_path(self):
        """Get CLASSIFIER_MODEL_PATH."""
        return self._config["CLASSIFIER_MODEL_PATH"]

    @property
    def label_encoder_path(self):
        """Get LABEL_ENCODER_PATH."""
        return self._config["LABEL_ENCODER_PATH"]

    @property
    def gemini_api_key_path(self):
        """Get GEMINI_API_KEY_PATH."""

        return self._config["GEMINI_API_KEY_PATH"]

    @property
    def openai_api_key_path(self):
        """Get OPENAI_API_KEY_PATH."""
        return self._config["OPENAI_API_KEY_PATH"]

    @property
    def google_credentials_path(self):
        """Get GOOGLE_CREDENTIALS_PATH."""
        return self._config["GOOGLE_CREDENTIALS_PATH"]

    @property
    def mistral_api_key_path(self):
        """Get MISTRAL_API_KEY_PATH."""
        return self._config["MISTRAL_API_KEY_PATH"]
    
    @property
    def gemini_api_key(self):
        """Get GEMINI_API_KEY (direct from env var)."""
        return self._config.get("GEMINI_API_KEY")
    
    @property
    def openai_api_key(self):
        """Get OPENAI_API_KEY (direct from env var)."""
        return self._config.get("OPENAI_API_KEY")
    
    @property
    def mistral_api_key(self):
        """Get MISTRAL_API_KEY (direct from env var)."""
        return self._config.get("MISTRAL_API_KEY")


# Create a global instance with relative path support
def get_config_file_path():
    """Get the config file path, supporting both absolute and relative paths."""
    # Try to find config.txt relative to this file
    config_dir = os.path.dirname(__file__)
    config_file = os.path.join(config_dir, "config.txt")
    
    # If config.txt doesn't exist, try scannerai_config.txt
    if not os.path.exists(config_file):
        config_file = os.path.join(config_dir, "scannerai_config.txt")
    
    # If still not found, return None (will use env vars only)
    if not os.path.exists(config_file):
        return None
    
    return config_file

config = Config(get_config_file_path())

# Usage example:
# from scannerai.config.config import config
# if config.debug_mode:
#     print("Debug mode is enabled")
