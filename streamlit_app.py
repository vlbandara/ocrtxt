"""Streamlit app entry point for deployment."""
import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root and src directory to the path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Import and run the main Streamlit app
from scripts.lcf_receipt_entry_streamlit import main

if __name__ == "__main__":
    main()

