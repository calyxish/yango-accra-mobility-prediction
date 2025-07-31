# Yango Accra Mobility Prediction

A machine learning project to predict ride travel times in Accra, Ghana using trip data and weather conditions.

## Project Structure

```
yango-accra-mobility-prediction/
├── data/                          # Data files (not tracked in git)
│   ├── Train.csv
│   ├── Test.csv
│   ├── Accra_weather.csv
│   ├── SampleSubmission.csv
│   └── VariableDefinitions.csv
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_train_model.ipynb
│   └── StarterNotebook.ipynb     # Start here!
├── scripts/                       # Python scripts
│   └── feature_engineering.py
├── outputs/                       # Model outputs and submissions
├── config.py                      # Configuration and file paths
├── utils.py                       # Utility functions
├── setup_check.py                 # Environment verification
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/calyxish/yango-accra-mobility-prediction.git
cd yango-accra-mobility-prediction
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n yango-env python=3.9
conda activate yango-env

# Or using venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Data
1. Download competition data from Zindi
2. Place CSV files in the `data/` directory:
   - `Train.csv`
   - `Test.csv` 
   - `Accra_weather.csv`
   - `SampleSubmission.csv`
   - `VariableDefinitions.csv`

### 5. Verify Setup
```bash
python setup_check.py
```

### 6. Start with Notebooks
```bash
jupyter notebook
```
Open `StarterNotebook.ipynb` to begin!

## Usage Guide

### For Beginners
1. **Start with**: `StarterNotebook.ipynb` - Basic implementation with RMSE ~4.4
2. **Learn from**: Comments and explanations in each cell
3. **Submit**: Generate `baseline_submission.csv` in outputs folder

### For Advanced Users
1. **EDA**: `01_eda_and_cleaning.ipynb` - Detailed data exploration
2. **Advanced Models**: `02_train_model.ipynb` - LightGBM, XGBoost, ensembles
3. **Feature Engineering**: Use `scripts/feature_engineering.py` for advanced features

## Key Features

- **Comprehensive Feature Engineering**: 30+ engineered features
- **Weather Integration**: Merge trip data with hourly weather
- **Multiple ML Algorithms**: Random Forest, LightGBM, XGBoost
- **Geospatial Features**: Distance calculations, location clustering
- **Time-based Features**: Rush hour, weekend, cyclical encoding
- **Automated Setup**: Environment verification and path management

## Project Components

### Configuration (`config.py`)
- Centralized file paths
- Constants and settings
- Data file validation

### Utilities (`utils.py`) 
- Data loading helpers
- Model evaluation functions
- Visualization utilities
- Submission file creation

### Feature Engineering (`scripts/feature_engineering.py`)
- DateTime feature extraction
- Distance calculations (Haversine)
- Location clustering (K-means)
- Weather feature engineering
- Interaction features

## Performance Benchmarks

| Model | Features | Validation RMSE |
|-------|----------|----------------|
| Baseline RF | 5 basic | ~4.4 |
| Enhanced RF | 15+ features | ~4.2 |
| LightGBM | 30+ features | ~4.1 |
| Ensemble | Multiple models | TBD |

## Development Workflow

### Adding New Features
1. Modify `scripts/feature_engineering.py`
2. Test in notebooks
3. Update feature lists
4. Retrain models

### Model Development
1. Experiment in `02_train_model.ipynb`
2. Use cross-validation
3. Track performance metrics
4. Save best models

### Collaboration
- **Pull latest changes**: `git pull origin main`
- **Create feature branch**: `git checkout -b feature-name`
- **Make changes and test**
- **Submit pull request**

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall packages
pip install -r requirements.txt --force-reinstall
```

**2. Data Files Not Found**
```bash
# Verify data directory structure
python setup_check.py
```

**3. Jupyter Kernel Issues**
```bash
# Install kernel
python -m ipykernel install --user --name yango-env
```

**4. Path Issues**
- Ensure you run notebooks from project root
- Use `config.py` for all file paths

### Getting Help
1. Check `setup_check.py` output
2. Review error messages carefully
3. Ensure virtual environment is activated
4. Verify all data files exist

## Contributing

### Guidelines
1. **Create feature branch** for new work
2. **Test thoroughly** before committing
3. **Update documentation** when needed
4. **Follow code style** (clean, commented)
5. **Don't commit data files** (use .gitignore)

### Code Standards
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and small
- Use meaningful variable names
- Handle errors gracefully

## Data Security

- **Data files are not tracked** in version control
- **Add data/ to .gitignore** to prevent accidental commits
- **Share only code and notebooks**, not datasets
- **Use relative paths** for cross-platform compatibility

## Next Steps

1. **Feature Engineering**: Add more sophisticated features
2. **Model Tuning**: Hyperparameter optimization
3. **Ensemble Methods**: Combine multiple models
4. **Cross-Validation**: Robust evaluation strategy
5. **Deployment**: Model serving and monitoring

