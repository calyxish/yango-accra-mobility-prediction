"""
Configuration file for file paths and constants
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file paths
TRAIN_FILE = DATA_DIR / "Train.csv"
TEST_FILE = DATA_DIR / "Test.csv"
WEATHER_FILE = DATA_DIR / "Accra_weather.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "SampleSubmission.csv"
VARIABLE_DEFINITIONS_FILE = DATA_DIR / "VariableDefinitions.csv"

# Model constants
RANDOM_SEED = 42
TARGET_COLUMN = "Target"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

def get_data_path(filename):
    """Get full path to data file"""
    return DATA_DIR / filename

def get_output_path(filename):
    """Get full path to output file"""
    return OUTPUTS_DIR / filename

def check_data_files():
    """Check if all required data files exist"""
    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        WEATHER_FILE,
        SAMPLE_SUBMISSION_FILE,
        VARIABLE_DEFINITIONS_FILE
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path.name)
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print(f"Please place data files in: {DATA_DIR}")
        return False
    
    print("All data files found successfully!")
    return True
