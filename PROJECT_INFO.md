#  Titanic Survival Prediction Project

## Overview
This project implements a complete machine learning pipeline to predict whether a passenger on the Titanic survived or not based on their personal information. It follows industry-standard practices and demonstrates senior data scientist level implementation.

## Project Structure
```
Titanic Project/
├── titanic_ml_project.py     # Professional ML pipeline
├── app.py                   # Flask web application  
├── run_ml_project.py        # Script to execute ML pipeline
├── run_website.py           # Script to run web interface
├── templates/               # HTML templates for web app
│   └── index.html
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
├── PROJECT_SUMMARY.md       # Detailed project summary
└── data/                   # Directory for datasets
```

## Key Features

### 1. ML Pipeline
- Complete exploratory data analysis (EDA) with visualizations
- Data preprocessing and cleaning
- Feature engineering (FamilySize, IsAlone, AgeGroups, etc.)
- Multiple algorithm implementation (Logistic Regression, Random Forest)
- Cross-validation and hyperparameter tuning
- Comprehensive model evaluation

### 2. Industry-Standard Practices
- Proper train-test splitting with stratification
- Handling of missing values
- Categorical encoding
- Feature scaling where appropriate
- Model validation techniques

### 3. Web Interface
- Interactive Flask-based web application
- Form for entering passenger details
- Real-time prediction results
- Responsive design

### 4. Code Quality
- Modular, well-documented code
- Clear separation of concerns
- Error handling and validation
- Reproducible results with fixed random seeds

## How to Run

### Option 1: ML Pipeline Only
```bash
pip install -r requirements.txt
python run_ml_project.py
```

### Option 2: Web Application
```bash
pip install -r requirements.txt
python run_website.py
```

## Technical Implementation Details

### Data Preprocessing
- Missing value imputation using median/mode
- Creation of new features (FamilySize, IsAlone, etc.)
- Encoding of categorical variables
- Outlier detection and handling

### Feature Engineering
- FamilySize = SibSp + Parch + 1
- IsAlone indicator variable
- Age group categorization
- Fare per person calculation

### Model Architecture
- Logistic Regression with regularization
- Random Forest with optimized hyperparameters
- Cross-validation for robust performance estimation

## Educational Value
This project demonstrates:
- End-to-end ML pipeline implementation
- Professional data science workflows
- Industry-standard coding practices
- Model evaluation and selection techniques
- Deployment considerations



