# Ski Snow Depth Predictor

## Project Overview
The **Ski Snow Depth Predictor** is a data-driven application designed to assist skiers in selecting the best resorts and months for skiing based on predicted snow depth. This application serves as a valuable tool for users interested in optimizing their ski trips by leveraging machine learning and historical weather data. The project provides insight into snow depth across multiple resorts in the Alps, helping users make data-driven decisions about their ski plans.

## Table of Contents
- [Project Overview](#project-overview)
- [Business Case](#business-case)
- [Hypotheses](#hypotheses)
- [Machine Learning Model](#machine-learning-model)
- [Data Collection and Processing](#data-collection-and-processing)
- [User Interface Design](#user-interface-design)
- [Model Evaluation and Performance](#model-evaluation-and-performance)
- [Technical Architecture](#technical-architecture)
- [Project Insights and Conclusions](#project-insights-and-conclusions)
- [Future Work](#future-work)
- [Deployment](#deployment)
- [File Structure](#file-structure)
- [Installation and Usage](#installation-and-usage)

## Business Case

**Objective**: The primary objective of the Ski Snow Depth Predictor is to forecast snow depth across various resorts in the Alps, meeting a business requirement for accessible, data-driven insights on optimal skiing conditions.

**Stakeholders**: Skiers, travel planners, and resort operators can benefit from accurate snow depth forecasts, enabling them to optimize trip planning, enhance user experience, and increase resort satisfaction rates.

### Business Requirements:
- Predict snow depth accurately for each resort and month within the operational ski season.
- Provide an interactive dashboard for users to explore data insights and predictions.
- Validate model predictions with statistical accuracy for reliability.

The project addresses these business requirements with the following data visualizations and ML tasks:

- **Snow Depth Distribution**: Visualizations for monthly and geographical snow depth.
- **Predictive Modeling**: A linear regression model predicts snow depth based on weather variables.

## Hypotheses

- **Hypothesis 1**: Resorts with higher altitudes experience greater snow depths.
  - **Validation**: Analyzed altitude and snow depth correlation with statistical measures.

- **Hypothesis 2**: Snow depth significantly varies between different resorts within the same month.
  - **Validation**: Box plots illustrate variance in snow depth by resort, validating resort-specific predictions.

- **Hypothesis 3**: Precipitation is a key predictor of snow depth across resorts.
  - **Validation**: Statistical analysis and feature importance rankings from the regression model demonstrate the influence of precipitation.

**Conclusion**: Hypotheses were validated through data analysis and correlation measures, offering valuable insights to stakeholders.

## Machine Learning Model

The model is a **Linear Regression model** designed to predict snow depth based on weather attributes. Key steps include:

- **Feature Engineering**: Added temperature average, precipitation sum, and month, country, and resort encoding.
- **Preprocessing Pipeline**: `StandardScaler` for numeric scaling and `OneHotEncoder` for categorical encoding.
- **Model Training and Hyperparameter Tuning**: Implemented hyperparameter tuning with cross-validation to optimize performance.

### Pipeline Components:
- **Preprocessor**: Encodes categorical variables and scales numeric ones.
- **Predictor**: Linear regression model.
- **Deployment**: The model and pipeline are saved and loaded via `pickle` for real-time predictions on Streamlit.

### Performance Metrics:
- **MAE**: Mean Absolute Error for validation and test sets.
- **RMSE**: Root Mean Squared Error for model precision.
- **R2 Score**: Model's accuracy in explaining the variance.

**Note**: Model outputs align with stakeholder requirements, meeting accuracy and performance thresholds.

## Data Collection and Processing

### Data Source
Collected from Meteostat, the data spans historical weather details and resort information, including temperature, precipitation, and snow depth.

### Data Cleaning and Transformation:
- **Handling Null Values**: Imputation and removal based on data integrity checks.
- **Feature Engineering**: Aggregated and processed features like average temperature, creating lag features, and encoded categorical variables.

### Data Storage:
- Cleaned and organized in a structured `dashboard_data.csv` file, adhering to CRISP-DM standards.

## User Interface Design

The dashboard is built with **Streamlit**, offering an interactive, responsive, and accessible user experience. Key features include:

### Dashboard Pages:
- **Home**: Overview and user guide.
- **Data Analysis**: Snow depth visualizations and correlation heatmaps.
- **Predictive Conditions by Resort**: Users select specific resorts and receive month-by-month snow depth predictions.
- **Predictive Conditions by Country**: Average snow depths across countries for a selected month.
- **Project Evaluation**: Residual analysis and model evaluation metrics.
- **About**: Project background, technologies used, and contributor details.

### Design Principles:
- **Hierarchy and Consistency**: Clear visual hierarchy, structured layout, and a coherent color scheme enhance navigation.
- **Accessibility**: Meets accessibility standards, offering contrast, font clarity, and easy navigation.
- **Interactive Components**: Widgets, selection boxes, and tooltips provide a smooth user experience.

## Model Evaluation and Performance

The model was assessed on both validation and test datasets, with visualizations in the project evaluation section:

- **Residual Analysis**: Visual residuals and error distribution.
- **Feature Importance**: (Optional) Feature ranking based on the model's regression coefficients.
- **Conclusions**: The model performs well within the set accuracy range, meeting project requirements.

## Technical Architecture

### File Structure
- `src/`: Contains code for data processing, modeling, and evaluation.
- `data/`: Data handling and preprocessing scripts.
- `models/`: Model training and evaluation scripts.
- `residual_analysis.py`: Analyzes residuals for model validation.
- `data/`: Stores raw and processed data, including `dashboard_data.csv`.
- `dashboard.py`: Main Streamlit application for deployment.

### Main Technologies
- **Python**: Core programming language.
- **Streamlit**: Dashboard framework.
- **scikit-learn**: Model development and preprocessing.
- **Pandas & NumPy**: Data handling and manipulation.
- **Matplotlib & Seaborn**: Data visualization.

## Project Insights and Conclusions

The model's predictive accuracy provides valuable insights for skiers and resort planners, helping to optimize decision-making regarding resort selection and trip timing. The validated hypotheses and interactive dashboard add transparency, promoting user engagement and trust.

## Future Work

- **Enhanced Model**: Testing with additional weather variables (e.g., wind speed, humidity) to improve predictions.
- **Expanded Coverage**: Adding more resorts for wider applicability.
- **User Feedback Loop**: Incorporating user feedback on model predictions to improve real-world applicability.

## Deployment

Deployed on **Heroku** using Streamlit, with all necessary files (`requirements.txt` etc.) for an automated deployment pipeline.

### Requirements for Local Deployment:
1. Install dependencies via `requirements.txt`.
2. Run `dashboard.py` using:
   streamlit run dashboard.py

## File Structure

.
├── app
├── dashboard.py
├── data
│   ├── external
│   ├── processed
│   │   ├── cds
│   │   │   ├── austrian_alps
│   │   │   │   ├── kitzbuhel
│   │   │   │   │   └── kitzbuhel_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   ├── solden
│   │   │   │   │   └── solden_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   └── st_anton
│   │   │   │       └── st_anton_cleaned_2024-10-28_13-30-56.csv
│   │   │   ├── dashboard_data.csv
│   │   │   ├── italian_alps
│   │   │   │   ├── cortina_d_ampezzo
│   │   │   │   │   └── cortina_d_ampezzo_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   ├── sestriere
│   │   │   │   │   └── sestriere_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   └── val_gardena
│   │   │   │       └── val_gardena_cleaned_2024-10-28_13-30-56.csv
│   │   │   ├── slovenian_alps
│   │   │   │   ├── kranjska_gora
│   │   │   │   │   └── kranjska_gora_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   ├── krvavec
│   │   │   │   │   └── krvavec_cleaned_2024-10-28_13-30-56.csv
│   │   │   │   └── mariborsko_pohorje
│   │   │   │       └── mariborsko_pohorje_cleaned_2024-10-28_13-30-56.csv
│   │   │   └── swiss_alps
│   │   │       └── st_moritz
│   │   │           └── st_moritz_cleaned_2024-10-28_13-30-56.csv
│   │   ├── modeling_data
│   │   │   ├── X_test.csv
│   │   │   ├── X_train.csv
│   │   │   ├── X_val.csv
│   │   │   ├── y_test.csv
│   │   │   ├── y_train.csv
│   │   │   └── y_val.csv
│   │   └── processed_data_for_modeling.csv
│   └── raw
│       └── cds
│           ├── austrian_alps
│           │   ├── kitzbuhel
│           │   │   └── kitzbuhel_meteostat_2024-10-28_10-48-10.csv
│           │   ├── solden
│           │   │   └── solden_meteostat_2024-10-28_10-48-12.csv
│           │   └── st._anton
│           │       └── st._anton_meteostat_2024-10-28_10-48-08.csv
│           ├── italian_alps
│           │   ├── cortina_d_ampezzo
│           │   │   └── cortina_d_ampezzo_meteostat_2024-10-28_10-48-17.csv
│           │   ├── sestriere
│           │   │   └── sestriere_meteostat_2024-10-28_10-48-19.csv
│           │   └── val_gardena
│           │       └── val_gardena_meteostat_2024-10-28_10-48-18.csv
│           ├── slovenian_alps
│           │   ├── kranjska_gora
│           │   │   └── kranjska_gora_meteostat_2024-10-28_10-48-20.csv
│           │   ├── krvavec
│           │   │   └── krvavec_meteostat_2024-10-28_10-48-23.csv
│           │   └── mariborsko_pohorje
│           │       └── mariborsko_pohorje_meteostat_2024-10-28_10-48-22.csv
│           └── swiss_alps
│               ├── st_moritz
│               │   └── st_moritz_meteostat_2024-10-28_10-48-14.csv
│               └── verbier
│                   └── verbier_meteostat_2024-10-28_10-48-15.csv
├── logs
│   └── modeling
│       └── residual_analysis
├── notebooks
│   ├── data
│   │   └── processed
│   │       └── modeling_data
│   │           ├── X_test.csv
│   │           ├── X_train.csv
│   │           ├── X_val.csv
│   │           ├── y_test.csv
│   │           ├── y_train.csv
│   │           └── y_val.csv
│   ├── data_cleaning.ipynb
│   ├── data_collection.ipynb
│   ├── evaluation.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   ├── Model_Evaluation_and_Interpretation.ipynb
│   ├── modeling.ipynb
│   └── src
│       └── models
├── README.md
├── requirements.txt
├── src
│   ├── data
│   │   ├── aggregate_dashboard_data.py
│   │   ├── cleaning.py
│   │   ├── fetch_data.py
│   │   ├── __init__.py
│   │   ├── processing.py
│   │   └── save_data.py
│   ├── eda
│   │   ├── __init__.py
│   │   ├── statistical_analysis.py
│   │   └── visualization.py
│   ├── features
│   │   ├── anomaly_detection.py
│   │   ├── feature_engineering.py
│   │   └── __init__.py
│   └── models
│       ├── __init__.py
│       ├── model_evaluation.py
│       ├── model_training.py
│       ├── plots
│       │   └── residuals
│       ├── residual_analysis.py
│       └── utils.py
└── tests
    ├── test_fetch_data.py
    ├── test_model_evaluation.py
    ├── test_model_training.py
    ├── test_processing.py
    ├── test_statistical_analysis.py
    ├── test_utils.py
    └── test_visualization.py

## Installation and Usage

1. Clone the repository
 - git clone https://github.com/yourusername/SkiSnow.git
 - cd SkiSnow
2. Install dependencies:
 - pip install -r requirements.txt
3. Run the application:
 - streamlit run dashboard.py
