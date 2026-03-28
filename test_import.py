import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

try:
    from src.utils import get_shap_explanation, check_image_quality
    print("SUCCESS: Both functions imported correctly!")
    print(f"check_image_quality location: {check_image_quality}")
except ImportError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")
