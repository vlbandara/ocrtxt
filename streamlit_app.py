"""Streamlit app entry point for deployment."""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Streamlit app
from scripts.lcf_receipt_entry_streamlit import main

if __name__ == "__main__":
    main()

