# AgriSmart Platform

**किसानों के लिए डिजिटल सलाहकार** - AI-Powered Agricultural Advisory Platform

## Problem Statement

Indian farmers face multiple challenges in modern agriculture:
- **Crop Selection Uncertainty**: Difficulty in determining optimal crops for specific soil and climate conditions
- **Disease Management**: Late detection of crop diseases leading to significant yield losses
- **Financial Planning**: Limited access to accurate loan estimation and agricultural financing information
- **Storage Losses**: Poor crop storage management resulting in post-harvest wastage
- **Information Gap**: Lack of centralized access to government schemes, market prices, and farming best practices

These challenges result in reduced productivity, economic losses, and missed opportunities for sustainable farming practices.

## Solution

AgriSmart is a comprehensive AI-powered agricultural advisory platform that addresses these challenges through intelligent automation and data-driven insights:

### Core Features

1. **Crop Recommendation System**
   - AI-powered suggestions based on soil parameters (pH, NPK levels)
   - Climate and location-based analysis
   - Optimal crop selection for maximum yield

2. **Disease Detection Module**
   - Instant disease diagnosis from leaf images
   - 95%+ accuracy using deep learning models
   - Early detection to prevent crop losses

3. **Loan Estimator**
   - Calculate precise investment requirements
   - Integration with NABARD loan schemes
   - Financial planning tools for farmers

4. **Storage Management Dashboard**
   - Smart crop storage monitoring
   - Automated alerts for optimal storage conditions
   - Shelf-life tracking to reduce wastage

5. **Knowledge Hub**
   - Daily farming tips and best practices
   - Government scheme information (PM-KISAN, PMFBY, PMDDKY)
   - Real-time market prices and forecasts
   - Weather updates and agricultural news

### Technical Architecture

- **Backend:** Python Flask for robust server-side processing
- **AI/ML Models:** TensorFlow and Scikit-learn for crop recommendation and disease detection
- **Database:** SQLite for efficient data management
- **Frontend:** Responsive design with soft pastel agricultural theme
- **Integration:** Government APIs and agricultural databases

## Project Structure

```
AgriSmartAI_Copy/
├── app.py                     # Main Flask application with routes
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── run.bat                    # Windows startup script
├── templates/                 # HTML templates (Jinja2)
│   ├── base.html             # Base template with navigation
│   ├── index.html            # Homepage with hero section
│   ├── crop_recommendation.html
│   ├── crop_results.html
│   ├── disease_detection.html
│   ├── disease_result.html
│   ├── loan_estimator.html
│   ├── loan_result.html
│   ├── storage_management.html
│   ├── storage_dashboard.html
│   ├── knowledge_hub.html
│   ├── 404.html              # Error pages
│   └── 500.html
├── static/
│   ├── css/
│   │   └── styles.css        # Soft pastel theme styling
│   ├── js/
│   │   └── main.js           # Frontend JavaScript
│   └── uploads/              # User uploaded images
├── services/                  # Business logic layer
│   ├── crop_ml_service.py    # Crop recommendation ML
│   ├── disease_detection_service.py
│   ├── disease_detection_trainer.py
│   ├── disease_service.py
│   └── loan_ml_service.py
├── models/                    # Trained ML models
│   ├── crop_recommendation/
│   ├── disease_detection/
│   │   ├── mobilenetv2_disease.h5
│   │   └── class_indices.json
│   └── loan_estimator/
├── utils/                     # Helper utilities
│   └── database.py           # SQLite database operations
├── datasets/                  # Training datasets
│   ├── Crop_Data.csv
│   ├── crop_production.csv
│   ├── Crop_recommendation.csv
│   ├── Train/                # Disease detection training images
│   ├── Test/
│   └── Validation/
└── data/
    └── agrismart.db          # SQLite database file
```

## Dataset Sources

All datasets used in this project are sourced from publicly available platforms:

### 1. Crop Data & Market Prices
- **File**: `Crop_Data.csv`
- **Source**: [Data.gov.in](https://data.gov.in) - Government of India Open Data Platform
- **Contains**: Market prices, commodity data from various states and districts

### 2. Crop Recommendation Dataset
- **File**: `Crop_recommendation.csv`
- **Source**: [Crop Recommendation System using LightGBM](https://www.kaggle.com/code/ysthehurricane/crop-recommendation-system-using-lightgbm)
- **Contains**: Soil parameters (N, P, K, pH) and climate data for 22 crops

### 3. Crop Production Dataset
- **File**: `crop_production.csv`
- **Source**: [Kaggle](https://www.kaggle.com)
- **Contains**: Historical crop production data by state, district, season, and crop type

### 4. Plant Disease Recognition Dataset
- **Files**: `Train/`, `Test/`, `Validation/` folders
- **Source**: [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
- **Note**: Due to size constraints, these image folders are not included in this repository. Download the dataset from Kaggle and place it in the `datasets/` folder to train the disease detection model.


### Design Philosophy

- **User-Centric Interface**: Intuitive design with soft pastel agricultural colors (Sage Green, Warm Cream, Sky Blue)
- **Mobile-First Approach**: Responsive design accessible on any device
- **Data-Driven Decisions**: Leveraging AI/ML for accurate predictions and recommendations
- **Accessibility**: Supporting multiple languages to reach farmers across India

## Impact

AgriSmart empowers farmers with modern technology to:
- Increase crop yields through informed decision-making
- Reduce losses from diseases and poor storage practices
- Access financial resources and government schemes efficiently
- Stay updated with market trends and agricultural knowledge

---
