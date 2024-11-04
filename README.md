# Ski Snow Depth Predictor

## Project Overview
The **Ski Snow Depth Predictor** provides skiers with predictive insights on snow depth across various resorts in the Alps. By leveraging historical weather data and machine learning, the tool forecasts snow depth, assisting users in choosing ideal skiing conditions based on specific months, countries, and resorts.

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
A **Linear Regression Model** is employed to predict snow depth based on historical monthly weather data:

- **Features**:
  - **Month**: One-hot encoded for categorical representation.
  - **Temperature Average and Precipitation Sum**: Used as continuous predictors.
  - **Country and Resort**: Represented categorically for location-specific insights.

- **Pipeline Steps**:
  - **Preprocessing**: Categorical features are one-hot encoded, while continuous features are scaled.
  - **Training**: Linear regression on training data with an evaluation on both validation and test sets.

The model outputs predictions based on resort, month, and selected region within the Alps, assisting users in selecting optimal skiing locations.

## Data Collection and Processing

### Sources
Data is sourced from **Meteostat**, covering weather metrics across resorts in the Alps from 1990 onwards. Relevant columns include `time`, `temperature_min`, `temperature_max`, `precipitation_sum`, and `snow_depth`.

### Processing Pipeline
- **Cleaning**: Raw data undergoes standardization, filling missing values with the mean, and categorical encoding.
- **VIF Calculation**: To address multicollinearity, features with high Variance Inflation Factor (VIF) values are iteratively removed.
- **Residual Analysis**: Plots are generated to assess model residuals, helping identify biases or prediction anomalies.

Data transformations are handled using the `src/data` and `src/features` modules, with utility functions for saving, loading, and logging.

## User Interface Design
The **Streamlit** dashboard features a user-friendly interface:

- **Home Page**: Introduction to the project and usage guidelines.
- **The Model**: Model explanation and interactive prediction based on user selections (resort, month, etc.).
- **Data Analysis**: Visual exploratory data analysis (EDA), including:
  - Snow depth distributions
  - Correlation heatmap for key features
  - Snow depth trends across months and countries
- **Predictive Conditions by Resort/Country**: Predict snow depths by selected resort or average snow depths by country.
- **Project Evaluation**: Model performance metrics, feature importance, and residual analysis plots.
- **About**: Information on project contributors and contact details.

### Design Principles:
- **Hierarchy and Consistency**: Clear visual hierarchy, structured layout, and a coherent color scheme enhance navigation.
- **Accessibility**: Meets accessibility standards, offering contrast, font clarity, and easy navigation.
- **Interactive Components**: Widgets, selection boxes, and tooltips provide a smooth user experience.

## Model Evaluation and Performance
Model performance is evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)** (with adjustments for cases with zero values in `y_true`)

Residual analysis functions generate plots showing residual distributions and residuals versus predictions, helping visualize model errors. The `src/models/residual_analysis.py` file provides tools for in-depth evaluation.

## Technical Architecture
This project is structured in a modular format with dedicated scripts for data processing, model training, evaluation, and visualization:

- **dashboard.py**: Primary dashboard script for Streamlit, implementing navigation, model loading, and data visualizations.
- **src/data**: Data collection, cleaning, and saving scripts.
- **src/features**: Feature engineering and anomaly detection.
- **src/models**: Model training, evaluation, and residual analysis.
- **src/eda**: Statistical analysis and visualization.

Logging is set up through `src/models/utils.py` to track the processing pipeline, including model and data-saving checkpoints.

## Main Technologies

- **Python**: The primary programming language for implementing data processing, machine learning models, and deployment. Python’s flexibility enables seamless integration across the various modules within the project.

- **Streamlit**: Used to build an interactive, user-friendly dashboard, allowing users to navigate through different sections, make selections, and view predictions. Streamlit facilitates rapid prototyping and real-time visualization of the model’s outputs.

- **scikit-learn**: Powers the machine learning aspects, including model training, evaluation, and preprocessing. scikit-learn’s robust set of tools for feature engineering, regression modeling, and metric calculations enables efficient experimentation and validation.

- **Pandas & NumPy**: Essential for data manipulation and numerical computation. Pandas handles complex data operations like merging, cleaning, and aggregation, while NumPy enhances speed and efficiency in handling numerical data, especially within the model training pipeline.

- **Matplotlib & Seaborn**: Provide comprehensive data visualization capabilities. Matplotlib and Seaborn are used extensively for visual exploratory data analysis (EDA) in the dashboard, creating histograms, heatmaps, line plots, and bar charts. These visualizations help users interpret snow depth trends, feature correlations, and model performance metrics.

- **Statsmodels**: Used specifically for calculating Variance Inflation Factor (VIF), ensuring features are not collinear and optimizing the model for reliability and interpretability.

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

```

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

```

## Installation and Usage

1. Clone the repository
 - git clone https://github.com/SamYatesSmith/SkiSnow.git
 - cd SkiSnow
2. Install dependencies:
 - pip install -r requirements.txt
3. Run the application:
 - streamlit run dashboard.py
