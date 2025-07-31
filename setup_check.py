#!/usr/bin/env python3
"""
Setup script for Yango Accra Mobility Prediction project
Run this script to verify the environment and data setup
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 
        'lightgbm', 'xgboost', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if project directories exist"""
    project_root = Path(__file__).parent
    required_dirs = ['data', 'notebooks', 'scripts', 'outputs']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"[OK] {dir_name}/ directory exists")
        else:
            print(f"[MISSING] {dir_name}/ directory missing")
            dir_path.mkdir(exist_ok=True)
            print(f"Created {dir_name}/ directory")
    
    return True

def check_data_files():
    """Check if data files are present"""
    from config import check_data_files
    return check_data_files()

def main():
    """Main setup verification"""
    print("Yango Accra Mobility Prediction - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Data Files", check_data_files)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            result = check_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"Error during {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("Setup verification completed successfully!")
        print("You can now run the notebooks.")
    else:
        print("Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
