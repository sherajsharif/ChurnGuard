import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page config with a more attractive theme
st.set_page_config(
    page_title="ChurnGuard - Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description with better formatting
st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #4CAF50; color: white; border-radius: 10px;'>
        <h1 style='margin: 0;'>🛡️ ChurnGuard</h1>
        <p style='margin: 10px 0 0 0;'>Customer Churn Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
        <p style='margin: 0; font-size: 16px;'>ChurnGuard helps businesses predict and analyze customer churn patterns using advanced machine learning. 
        Upload your customer data to get instant predictions and insights to improve your retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with better styling
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #4CAF50; color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='margin: 0;'>Upload Data</h3>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;'>
            <p style='margin: 0; font-size: 14px;'>Supported file format: CSV</p>
        </div>
    """, unsafe_allow_html=True)

def process_data(df):
    # Create a copy of the dataframe to avoid modifications to the original
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert binary categorical variables
    if 'gender' in df.columns:
        df["gender"] = df["gender"].map({'Male':0, 'Female':1})
    
    # Label encoding for binary columns
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] 
                   and df[col].nunique() == 2]
    
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encoding for categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype not in [int, float] 
                and df[col].nunique() > 2
                and col not in ['customerID', 'Churn']]  # Exclude these columns from encoding
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Standardization of numerical columns
    num_cols = [col for col in df.columns if df[col].dtype in [int, float] 
                and col not in ['customerID', 'Churn']]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def train_model(df):
    # Prepare data for modeling
    if "Churn" not in df.columns:
        st.error("Error: 'Churn' column not found in the dataset!")
        return None, None
    
    # Convert Churn to numeric if it's not already
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({'No': 0, 'Yes': 1})
    
    y = df["Churn"]
    
    # Drop columns safely
    columns_to_drop = ['Churn']
    if 'customerID' in df.columns:
        columns_to_drop.append('customerID')
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
    
    # Train model
    model = RandomForestClassifier(random_state=46)
    model.fit(X_train, y_train)
    
    # Calculate and display accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    st.write(f"Training Accuracy: {train_accuracy:.2%}")
    st.write(f"Testing Accuracy: {test_accuracy:.2%}")
    
    return model, X.columns

def main():
    if uploaded_file is not None:
        try:
            # Load data with progress indicator
            with st.spinner('Loading and processing data...'):
                df = pd.read_csv(uploaded_file)
            
            st.success("✅ Data successfully loaded!")
            
            # Show raw data in an expandable section
            with st.expander("📊 View Raw Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Check required columns
            required_columns = ['Churn']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Error: Missing required columns: {', '.join(missing_columns)}")
                st.info("ℹ️ Your dataset must contain a 'Churn' column with values indicating whether customers churned (e.g., 'Yes'/'No' or 1/0)")
                return
            
            # Data processing
            with st.spinner('Processing data...'):
                processed_df = process_data(df)
            
            # Train model
            with st.spinner('Training model...'):
                model, feature_names = train_model(processed_df)
            
            if model is not None and feature_names is not None:
                # Save model
                with open('churn_model.pkl', 'wb') as file:
                    pickle.dump(model, file)
                
                # Model Analysis with better visualization
                st.markdown("""
                    <div style='text-align: center; padding: 10px; background-color: #4CAF50; color: white; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='margin: 0;'>Model Analysis</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Feature importance with Plotly
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(feature_importance.head(10), 
                           x='importance', 
                           y='feature',
                           orientation='h',
                           title='Top 10 Most Important Features',
                           color='importance',
                           color_continuous_scale='Viridis')
                
                fig.update_layout(
                    height=500,
                    width=800,
                    showlegend=False,
                    xaxis_title="Importance",
                    yaxis_title="Feature"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Churn Distribution with Plotly
                churn_counts = df['Churn'].value_counts()
                fig = px.pie(values=churn_counts.values,
                           names=churn_counts.index,
                           title='Churn Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button with better styling
                st.markdown("""
                    <div style='text-align: center; margin: 20px 0;'>
                        <h4>Download Trained Model</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                with open('churn_model.pkl', 'rb') as file:
                    st.download_button(
                        label="⬇️ Download Model",
                        data=file,
                        file_name="churn_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please make sure your CSV file has the correct format and required columns.")
            st.info("""
            ℹ️ Required format for the dataset:
            - Must be a CSV file
            - Must contain a 'Churn' column (values can be 'Yes'/'No' or 1/0)
            - Should contain customer features (both numerical and categorical)
            - Should not contain any special characters in column names
            """)
            
    else:
        st.info("ℹ️ Please upload a CSV file to begin the analysis.")
        st.markdown("""
        ### Sample Dataset Format
        Your dataset should look something like this:
        
        | customerID | gender | SeniorCitizen | Partner | ... | Churn |
        |-----------|--------|---------------|---------|-----|-------|
        | 1234      | Male   | 0            | Yes     | ... | No    |
        | 5678      | Female | 1            | No      | ... | Yes   |
        
        The only required column is 'Churn', which should indicate whether the customer churned (Yes/No or 1/0).
        """)
        
if __name__ == "__main__":
    main() 