import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import calendar

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.models.residual_analysis import perform_residual_analysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------

st.set_page_config(
    page_title="❄️ Ski Snow Depth Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------
# Define Paths
# -----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')
MODEL_FILE = 'linear_regression_model.joblib'
DASHBOARD_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'cds')
DASHBOARD_DATA_FILE = 'dashboard_data.csv'

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

@st.cache_resource
def load_model(model_filename):
    """
    Loads the pre-trained machine learning model.
    """
    model_path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    # Use joblib to load the model
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_dashboard_data(filepath):
    """
    Loads the dashboard-specific processed data.
    """
    if not os.path.exists(filepath):
        st.error(f"Dashboard data file not found: {filepath}")
        st.stop()
    dashboard_data = pd.read_csv(filepath)
    return dashboard_data

def prepare_prediction_input(user_selection, dashboard_data, model):
    """
    Prepares the input DataFrame based on user selections for prediction.
    """
    # Filter data based on country and resort
    filtered_data = dashboard_data[
        (dashboard_data['country'] == user_selection['country']) &
        (dashboard_data['resort'] == user_selection['resort']) &
        (dashboard_data['month'] == user_selection['month'])
    ]

    if filtered_data.empty:
        st.warning("No data available for the selected combination. Please adjust your selections.")
        return None

    # Assuming the model expects 'temperature_avg' and 'precipitation_sum'
    # Here, you can set default values or compute based on historical data
    # For demonstration, we'll use mean values
    temp_avg = filtered_data['temperature_avg'].mean()
    precip_sum = filtered_data['precipitation_sum'].mean()

    input_data = pd.DataFrame({
        'month': [user_selection['month']],
        'country': [user_selection['country']],
        'resort': [user_selection['resort']],
        'temperature_avg': [temp_avg],
        'precipitation_sum': [precip_sum]
    })

    # One-Hot Encoding for 'month', 'country', 'resort' as per model's training
    input_encoded = pd.get_dummies(input_data, columns=['month', 'country', 'resort'], drop_first=True)

    # Align with model's expected features
    # Extract feature names from the preprocessor step in the pipeline
    # Assuming the model is a pipeline with named_steps including 'preprocessor' and 'regressor'
    # If not, adjust accordingly
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps['preprocessor']
        model_features = preprocessor.get_feature_names_out()
    else:
        model_features = model.feature_names_in_

    for feature in model_features:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0  # Add missing features

    input_encoded = input_encoded[model_features]  # Ensure correct order

    return input_encoded

def display_metrics(y_true, y_pred, model_name, dataset_name):
    """
    Displays evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    valid_mask = y_true != 0
    if valid_mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[valid_mask], y_pred[valid_mask]) * 100
    else:
        mape = np.nan  # Undefined if no valid actuals

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    }
    st.write(f"**{model_name} - {dataset_name} Set:**")
    st.table(pd.DataFrame(metrics, index=[0]))

# -----------------------------------------------------------
# Load Model and Data
# -----------------------------------------------------------
lr_model = load_model(MODEL_FILE)
dashboard_data = load_dashboard_data(os.path.join(DASHBOARD_DATA_DIR, DASHBOARD_DATA_FILE))

# -----------------------------------------------------------
# Navigation Menu
# -----------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", [
    "Home",
    "The Model",
    "Data Analysis",
    "Predictive Conditions by Resort",
    "Predictive Conditions by Country",
    "Project Evaluation and Analysis",
    "About"
])

# -----------------------------------------------------------
# Define Pages
# -----------------------------------------------------------

def home_page():
    st.title("🏔️ Welcome to the Ski Snow Depth Predictor")
    st.write("""
    **Purpose:** This dashboard provides skiers with predictive insights on snow depth across various resorts in the Alps. Plan your skiing trips by selecting your desired month, country, and resort to receive accurate snow depth predictions.
    
    **How It Works:** Utilizing historical weather data and machine learning algorithms, our model forecasts snow depth to help you choose the best skiing conditions.
    
    **User Guide:**
    - **Home:** Introduction and overview.
    - **The Model:** Details about the machine learning model and interactive prediction tool.
    - **Data Analysis:** Exploratory data analysis and visualizations.
    - **Predictive Conditions by Resort/Country:** Make predictions based on your selections.
    - **Project Evaluation and Analysis:** Assess model performance and project insights.
    - **About:** Information about the project and its creators.
    """)

def the_model_page():
    st.title("🤖 The Model")
    st.write("""
    **Machine Learning Model:** Linear Regression
    
    **Objective:** To predict snow depth (in cm) based on selected month, Alps region, resort, average temperature, and precipitation.
    
    **Features Used:**
    - **Month:** Encoded as categorical variables.
    - **Alps Region:** Encoded as categorical variables.
    - **Resort:** Encoded as categorical variables.
    - **Temperature Average:** Continuous variable.
    - **Precipitation Sum:** Continuous variable.
    
    **Model Pipeline:**
    1. **Encoding:** One-Hot Encoding for categorical features.
    2. **Scaling:** Standardization of continuous features.
    3. **Regression:** Linear Regression algorithm for prediction.
    
    **Performance Metrics:**
    - **Validation Set:**
      - MAE: _[Value]_
      - RMSE: _[Value]_
      - MAPE: _[Value]_
    - **Test Set:**
      - MAE: _[Value]_
      - RMSE: _[Value]_
      - MAPE: _[Value]_
    
    **Interpretation:** The model demonstrates _[describe performance]_, indicating _[insights about accuracy]_. Further improvements could involve _[suggestions like feature engineering, model selection, etc.]_.
    """)
    
    st.write("---")
    st.header("🔍 Predict the Best Resort for Your Skiing Trip")
    
    # 1. User Input: Select Month (In-Season Only)
    # Extract unique months where 'is_operating_season' is True
    in_season_months = dashboard_data.loc[dashboard_data['is_operating_season'] == True, 'month'].unique()
    
    # Remove NaN values and ensure uniqueness
    in_season_months = [month for month in in_season_months if pd.notna(month)]
    
    # Sort months in calendar order
    sorted_months = sorted(in_season_months, key=lambda x: list(calendar.month_name).index(x))
    
    selected_month = st.selectbox("Select Month:", sorted_months)
    
    # 2. User Input: Select Alps Region
    alps_regions = sorted(dashboard_data['alps'].dropna().unique())
    selected_alps = st.selectbox("Select Alps Region:", alps_regions)
    
    # 3. User Input: Select Resorts (Optional)
    # Option to select all resorts or specific ones
    resorts = sorted(dashboard_data[
        (dashboard_data['alps'] == selected_alps) &
        (dashboard_data['month'] == selected_month) &
        (dashboard_data['is_operating_season'] == True)
    ]['resort'].unique())
    
    selected_resorts = st.multiselect(
        "Select Resorts (leave blank to include all):",
        resorts,
        default=resorts  # Default to all resorts in the selected region and month
    )
    
    # If no resorts selected, include all
    if not selected_resorts:
        selected_resorts = resorts
    
    # Prepare DataFrame for Predictions
    prediction_inputs = []
    for resort in selected_resorts:
        # Filter data for the selected month and resort within operating season
        resort_data = dashboard_data[
            (dashboard_data['resort'] == resort) &
            (dashboard_data['month'] == selected_month) &
            (dashboard_data['is_operating_season'] == True)
        ]
        
        if resort_data.empty:
            st.warning(f"No historical data available for {resort} in {selected_month}.")
            continue
        
        # Compute average temperature and precipitation
        # Adjust the computation based on your data's actual structure
        temp_avg = (resort_data['temperature_min'].mean() + resort_data['temperature_max'].mean()) / 2
        precip_sum = resort_data['precipitation_sum'].mean()
        
        # Create input dictionary
        input_dict = {
            'month': selected_month,
            'alps': selected_alps,
            'resort': resort,
            'temperature_avg': temp_avg,
            'precipitation_sum': precip_sum
        }
        prediction_inputs.append(input_dict)
    
    if not prediction_inputs:
        st.warning("No data available to make predictions. Please adjust your selections.")
        return
    
    input_df = pd.DataFrame(prediction_inputs)
    
    # One-Hot Encode categorical variables as per model's training
    input_encoded = pd.get_dummies(input_df, columns=['month', 'alps', 'resort'], drop_first=True)
    
    # Align with model's expected features
    # Retrieve model's feature names
    if hasattr(lr_model, 'named_steps'):
        preprocessor = lr_model.named_steps['preprocessor']
        model_features = preprocessor.get_feature_names_out()
    else:
        model_features = lr_model.feature_names_in_
    
    # Add missing features with default value 0
    for feature in model_features:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0
    
    # Ensure correct order of features
    input_encoded = input_encoded[model_features]
    
    # Make Predictions
    predictions = lr_model.predict(input_encoded)
    input_df['Predicted Snow Depth (cm)'] = predictions
    
    # Identify the Best Resort
    best_resort = input_df.loc[input_df['Predicted Snow Depth (cm)'].idxmax()]
    
    # Display Predictions
    st.subheader("### Predicted Snow Depths:")
    st.table(
        input_df[['resort', 'Predicted Snow Depth (cm)']]
        .sort_values(by='Predicted Snow Depth (cm)', ascending=False)
        .reset_index(drop=True)
    )
    
    # Highlight Best Resort
    st.success(
        f"**Best Resort for {selected_month} in {selected_alps}: {best_resort['resort']} with predicted snow depth of {best_resort['Predicted Snow Depth (cm)']:.2f} cm**"
    )
    
    # Plotting Predicted Snow Depths
    st.subheader("### Predicted Snow Depths by Resort")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(
        x='resort',
        y='Predicted Snow Depth (cm)',
        data=input_df.sort_values(by='Predicted Snow Depth (cm)', ascending=False),
        palette='viridis',
        ax=ax
    )
    ax.set_title(f"Predicted Snow Depths in {selected_month}")
    ax.set_xlabel("Resort")
    ax.set_ylabel("Snow Depth (cm)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    

def data_analysis_page():
    st.title("📊 Data Analysis")
    st.write("### Exploratory Data Analysis")
    
    # Example Plot 1: Snow Depth Distribution
    st.subheader("Snow Depth Distribution")
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.histplot(dashboard_data['snow_depth'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel("Snow Depth (cm)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Snow Depth Across Resorts")
    st.pyplot(fig1)
    
    # Example Plot 2: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    # Ensure required columns exist
    required_columns = ['temperature_avg', 'precipitation_sum', 'snow_depth']
    missing_columns = [col for col in required_columns if col not in dashboard_data.columns]
    if missing_columns:
        st.error(f"Missing columns in data: {missing_columns}")
    else:
        corr = dashboard_data[required_columns].corr()
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title("Correlation Between Features")
        st.pyplot(fig2)
    
    # Additional Plots as per marking criteria
    st.subheader("Snow Depth by Month")
    if 'month' in dashboard_data.columns and 'snow_depth' in dashboard_data.columns:
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.boxplot(x='month', y='snow_depth', data=dashboard_data, palette='Set3', ax=ax3)
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Snow Depth (cm)")
        ax3.set_title("Snow Depth Distribution Across Months")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
    else:
        st.error("Required columns for 'Snow Depth by Month' plot are missing.")
    
    st.subheader("Snow Depth by Country")
    if 'country' in dashboard_data.columns and 'snow_depth' in dashboard_data.columns:
        fig4, ax4 = plt.subplots(figsize=(10,6))
        sns.boxplot(x='country', y='snow_depth', data=dashboard_data, palette='Set2', ax=ax4)
        ax4.set_xlabel("Country")
        ax4.set_ylabel("Snow Depth (cm)")
        ax4.set_title("Snow Depth Distribution Across Countries")
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    else:
        st.error("Required columns for 'Snow Depth by Country' plot are missing.")

def predictive_conditions_resort_page():
    st.title("🎯 Predictive Conditions by Resort")
    st.write("""
    Select a month, country, and resort to receive a predicted snow depth.
    """)
    
    # User Selections
    months = sorted(dashboard_data['month'].unique())
    countries = sorted(dashboard_data['country'].unique())
    selected_country = st.selectbox("Select Country:", countries)
    resorts = sorted(dashboard_data[dashboard_data['country'] == selected_country]['resort'].unique())
    selected_resort = st.selectbox("Select Resort:", resorts)
    selected_month = st.selectbox("Select Month:", months)
    
    # Prepare input
    user_selection = {
        'country': selected_country,
        'resort': selected_resort,
        'month': selected_month
    }
    
    input_encoded = prepare_prediction_input(user_selection, dashboard_data, lr_model)
    
    if input_encoded is not None:
        try:
            prediction = lr_model.predict(input_encoded)
            st.success(f"**Predicted Snow Depth at {selected_resort} in {selected_month}: {prediction[0]:.2f} cm**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    
    # Historical Snow Depth Plot for Selected Resort
    st.subheader(f"Historical Snow Depth at {selected_resort}")
    resort_data = dashboard_data[dashboard_data['resort'] == selected_resort]
    if not resort_data.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(x='month', y='snow_depth', data=resort_data, marker='o', ax=ax)
        ax.set_title(f"Snow Depth Over Months at {selected_resort}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Snow Depth (cm)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning(f"No historical data available for {selected_resort}.")

def predictive_conditions_country_page():
    st.title("🌍 Predictive Conditions by Country")
    st.write("""
    Select a month and country to receive average predicted snow depth across all resorts in that country.
    """)
    
    # User Selections
    months = sorted(dashboard_data['month'].unique())
    countries = sorted(dashboard_data['country'].unique())
    selected_country = st.selectbox("Select Country:", countries)
    selected_month = st.selectbox("Select Month:", months)
    
    # Prepare input for all resorts in the selected country
    country_resorts = sorted(dashboard_data[dashboard_data['country'] == selected_country]['resort'].unique())
    
    predictions = []
    for resort in country_resorts:
        user_selection = {
            'country': selected_country,
            'resort': resort,
            'month': selected_month
        }
        input_encoded = prepare_prediction_input(user_selection, dashboard_data, lr_model)
        if input_encoded is not None:
            try:
                pred = lr_model.predict(input_encoded)[0]
                predictions.append({'resort': resort, 'Predicted Snow Depth (cm)': pred})
            except:
                predictions.append({'resort': resort, 'Predicted Snow Depth (cm)': np.nan})
        else:
            predictions.append({'resort': resort, 'Predicted Snow Depth (cm)': np.nan})
    
    prediction_df = pd.DataFrame(predictions).dropna()
    
    if not prediction_df.empty:
        average_snow_depth = prediction_df['Predicted Snow Depth (cm)'].mean()
        st.success(f"**Average Predicted Snow Depth in {selected_country} during {selected_month}: {average_snow_depth:.2f} cm**")
        
        # Plot Average Snow Depth by Resort
        st.subheader(f"Snow Depth Predictions for {selected_country} Resorts in {selected_month}")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x='resort', y='Predicted Snow Depth (cm)', data=prediction_df.sort_values(by='Predicted Snow Depth (cm)', ascending=False), palette='viridis', ax=ax)
        ax.set_title(f"Predicted Snow Depths at {selected_country} Resorts in {selected_month}")
        ax.set_xlabel("Resort")
        ax.set_ylabel("Snow Depth (cm)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No predictions available for the selected criteria.")

def project_evaluation_page():
    st.title("📈 Project Evaluation and Analysis")
    st.write("""
    **Model Performance:**
    - **Validation Set Metrics:**
      - MAE: _[Value]_
      - RMSE: _[Value]_
      - MAPE: _[Value]_
    - **Test Set Metrics:**
      - MAE: _[Value]_
      - RMSE: _[Value]_
      - MAPE: _[Value]_
    
    **Residual Analysis:**
    - Visual representations of residuals to assess model accuracy and identify potential biases.
    
    **Feature Importance:**
    - Insights into which features significantly impact snow depth predictions.

    **Conclusion:**
    - Summary of model effectiveness in meeting business requirements.
    - Recommendations for future improvements or alternative approaches.
    """)
    
    st.subheader("Model Performance Metrics")
    
    # Load evaluation data
    modeling_data_dir = os.path.join(BASE_DIR, 'data', 'processed', 'modeling_data')
    X_val = pd.read_csv(os.path.join(modeling_data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(modeling_data_dir, 'y_val.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(modeling_data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(modeling_data_dir, 'y_test.csv')).squeeze()
    
    # Define the check_invalid_values function with 3 parameters
    def check_invalid_values(y, dataset_name, X):
        """
        Checks for NaN and Inf values in the target variable and removes corresponding rows from X.

        Parameters:
        - y (pd.Series or np.array): True target values.
        - dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test').
        - X (pd.DataFrame): Feature set corresponding to y.

        Returns:
        - y_clean (pd.Series or np.array): Cleaned target values without NaN or Inf.
        - X_clean (pd.DataFrame): Cleaned feature set corresponding to y_clean.
        """
        num_nan = y.isna().sum()
        num_inf = np.isinf(y).sum()
        if num_nan > 0 or num_inf > 0:
            st.error(f"{dataset_name} Set contains {num_nan} NaN(s) and {num_inf} Inf(s). These rows will be removed for residual analysis.")
            # Remove NaNs and Infs
            valid_mask = (~y.isna()) & (~np.isinf(y))
            return y[valid_mask], X[valid_mask]
        return y, X
    
    # Validate and clean Validation Set
    y_val_clean, X_val_clean = check_invalid_values(y_val, "Validation", X_val)
    
    # Validate and clean Test Set
    y_test_clean, X_test_clean = check_invalid_values(y_test, "Test", X_test)
    
    # Display Metrics for Validation Set
    st.subheader("Validation Set Metrics")
    val_predictions = lr_model.predict(X_val_clean)
    display_metrics(y_val_clean, val_predictions, "Linear Regression", "Validation")
    
    # Display Metrics for Test Set
    st.subheader("Test Set Metrics")
    test_predictions = lr_model.predict(X_test_clean)
    display_metrics(y_test_clean, test_predictions, "Linear Regression", "Test")
    
    # Residual Analysis for Validation Set
    st.subheader("Residual Analysis - Validation Set")
    try:
        perform_residual_analysis(
            model=lr_model,
            X=X_val_clean,
            y=y_val_clean,
            dataset_name="Validation",
            save_dir=os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis')
        )
        
        # Display Residuals vs Predicted Plot
        residuals_vs_predicted_val = os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis', 'residuals_vs_predicted_validation.png')
        if os.path.exists(residuals_vs_predicted_val):
            st.image(residuals_vs_predicted_val, caption="Residuals vs Predicted Values (Validation Set)", use_column_width=True)
        else:
            st.warning("Residuals vs Predicted plot for Validation Set not found.")
        
        # Display Residuals Distribution Plot
        residuals_dist_val = os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis', 'residuals_distribution_validation.png')
        if os.path.exists(residuals_dist_val):
            st.image(residuals_dist_val, caption="Distribution of Residuals (Validation Set)", use_column_width=True)
        else:
            st.warning("Residuals Distribution plot for Validation Set not found.")
        
    except Exception as e:
        st.error(f"Error during residual analysis for Validation Set: {e}")
    
    # Residual Analysis for Test Set
    st.subheader("Residual Analysis - Test Set")
    try:
        perform_residual_analysis(
            model=lr_model,
            X=X_test_clean,
            y=y_test_clean,
            dataset_name="Test",
            save_dir=os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis')
        )
        
        # Display Residuals vs Predicted Plot
        residuals_vs_predicted_test = os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis', 'residuals_vs_predicted_test.png')
        if os.path.exists(residuals_vs_predicted_test):
            st.image(residuals_vs_predicted_test, caption="Residuals vs Predicted Values (Test Set)", use_column_width=True)
        else:
            st.warning("Residuals vs Predicted plot for Test Set not found.")
        
        # Display Residuals Distribution Plot
        residuals_dist_test = os.path.join(BASE_DIR, 'logs', 'modeling', 'residual_analysis', 'residuals_distribution_test.png')
        if os.path.exists(residuals_dist_test):
            st.image(residuals_dist_test, caption="Distribution of Residuals (Test Set)", use_column_width=True)
        else:
            st.warning("Residuals Distribution plot for Test Set not found.")
        
    except Exception as e:
        st.error(f"Error during residual analysis for Test Set: {e}")
    

def about_page():
    st.title("ℹ️ About")
    st.write("""
    **Project Overview:**
    This dashboard leverages historical weather data and machine learning algorithms to predict snow depth at various ski resorts in the Alps. It empowers skiers with data-driven insights to plan their trips effectively.

    **Technologies Used:**
    - **Python:** Programming language for data processing and model building.
    - **Streamlit:** Framework for building interactive dashboards.
    - **Pandas & NumPy:** Data manipulation and analysis.
    - **Scikit-Learn:** Machine learning library for model training and evaluation.
    - **Matplotlib & Seaborn:** Data visualization.

    **Project Contributors:**
    - **Your Name:** Sam Yates-Smith

    **Contact Information:**
    - **Email:** samyatessmith@yahoo.co.uk

    """)

# -----------------------------------------------------------
# Page Routing
# -----------------------------------------------------------

if menu == "Home":
    home_page()
elif menu == "The Model":
    the_model_page()
elif menu == "Data Analysis":
    data_analysis_page()
elif menu == "Predictive Conditions by Resort":
    predictive_conditions_resort_page()
elif menu == "Predictive Conditions by Country":
    predictive_conditions_country_page()
elif menu == "Project Evaluation and Analysis":
    project_evaluation_page()
elif menu == "About":
    about_page()
else:
    st.error("Page not found.")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("© 2024 Ski Snow Depth Predictor")
