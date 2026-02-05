# heart_disease_prediction_custom.py
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="CardioCare AI - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Custom CSS with red button styling
st.markdown("""
<style>
    /* Main colors */
    :root {
        --primary: #aa4848;
        --secondary: #9d50bb;
        --text: #333333;
        --light-text: #6c757d;
        --background: #f8f9fa;
        --card-bg: #ffffff;
    }

    /* Force red button styling */
    div.stButton > button:first-child,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #ff0000 !important;
        color: white !important;
        border: 2px solid #cc0000 !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        margin: 20px 0 !important;
        width: 100% !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:first-child:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #cc0000 !important;
        transform: translateY(-1px) !important;
    }

    /* Ensure form elements are visible */
    div[data-testid="stForm"] {
        position: relative !important;
        z-index: 1 !important;
    }

    /* Rest of your styling */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Arial', sans-serif;
    }

    h1 {
        color: var(--primary) !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: var(--primary) !important;
        font-weight: 600;
        margin-top: 1.5rem;
    }

    h3 {
        color: var(--primary) !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] {
        background: var(--card-bg) !important;
        border-right: 1px solid #e0e0e0;
    }

    [data-testid="stSidebar"] h2 {
        color: var(--primary) !important;
    }

    [data-testid="stSidebar"] p {
        color: var(--light-text) !important;
    }

    .card {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: none;
    }

    .stForm {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border: 1px solid #ced4da !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
    }

    p, .stMarkdown {
        color: var(--text) !important;
        line-height: 1.6;
    }

    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('heart_disease_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


model, scaler = load_model()


# Prediction function
def predict_heart_disease(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]

        return {
            'prediction': prediction[0],
            'probability': max(probability),
            'class': 'Positive' if prediction[0] == 1 else 'Negative'
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Main app function
def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <h2 style="color:#6e48aa; margin-bottom:0.5rem;">CardioCare AI</h2>
            <p style="color:#6c757d; font-size:0.9rem;">Heart Disease Prediction System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.9rem; color:#6c757d; margin-bottom:1.5rem;">
            This application predicts the likelihood of heart disease using machine learning.
            For medical emergencies, consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)

        app_mode = st.radio(
            "Navigation",
            ["Home", "Prediction", "Data Analysis", "About"],
            label_visibility="collapsed"
        )

    # Home page
    if app_mode == "Home":
        st.title("CardioCare AI")
        st.markdown("## Heart Disease Prediction System")

        st.markdown("""
        <div class="card">
            <p>Welcome to CardioCare AI, an advanced system for predicting heart disease risk 
            based on clinical parameters. Use the navigation menu to get started.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <h3>How to Use</h3>
                <ol>
                    <li>Go to <b>Prediction</b> section</li>
                    <li>Enter patient details</li>
                    <li>Click 'Predict' for risk assessment</li>
                    <li>View detailed analysis</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h3>Disclaimer</h3>
                <p>This tool is for informational purposes only and should not replace 
                professional medical advice. Always consult a healthcare provider for 
                medical concerns.</p>
            </div>
            """, unsafe_allow_html=True)

    # Prediction page
    elif app_mode == "Prediction":
        st.title("Heart Disease Risk Predictor")

        with st.form("prediction_form"):
            st.markdown("""
            <div style="margin-bottom:1.5rem;">
                Please enter the patient's clinical parameters:
            </div>
            """, unsafe_allow_html=True)

            # Form layout with two columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Personal Information")
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
                sex = st.selectbox("Sex", ["Female", "Male"])
                cp = st.selectbox("Chest Pain Type", [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ])
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
                restecg = st.selectbox("Resting ECG Results", [
                    "Normal",
                    "ST-T Wave Abnormality",
                    "Left Ventricular Hypertrophy"
                ])

            with col2:
                st.markdown("### Exercise Test Results")
                thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=220, value=150)
                exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0,
                                          step=0.1, format="%.2f")
                slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ])
                ca = st.selectbox("Number of Major Vessels (0-3)", ["0", "1", "2", "3"])
                thal = st.selectbox("Thalassemia", [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect"
                ])

            submitted = st.form_submit_button("Predict Heart Disease Risk")

            if submitted:
                with st.spinner('Analyzing patient data...'):
                    # Process inputs
                    input_data = {
                        'age': age,
                        'sex': 1 if sex == "Male" else 0,
                        'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': 1 if fbs == "Yes" else 0,
                        'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
                        'thalach': thalach,
                        'exang': 1 if exang == "Yes" else 0,
                        'oldpeak': oldpeak,
                        'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
                        'ca': int(ca),
                        'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
                    }

                    # Make prediction
                    result = predict_heart_disease(input_data)

                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    if result is not None:
                        if result['prediction'] == 1:
                            st.error(f"**High Risk of Heart Disease** (Probability: {result['probability']:.2%})")
                            st.warning("Recommendation: Please consult a cardiologist immediately.")
                        else:
                            st.success(f"**Low Risk of Heart Disease** (Probability: {1 - result['probability']:.2%})")
                            st.info("Recommendation: Maintain a healthy lifestyle with regular checkups.")

    # Data Analysis page
    elif app_mode == "Data Analysis":
        st.title("Data Analysis")
        st.write("This section shows analysis of the heart disease dataset.")

        # Load dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(url, names=column_names, na_values='?')
        df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

        with st.expander("View Dataset"):
            st.dataframe(df.head())

        with st.expander("Dataset Statistics"):
            st.dataframe(df.describe())

        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['age'], kde=True, ax=ax, color='#6e48aa')
            st.pyplot(fig)

        with col2:
            st.markdown("#### Heart Disease by Gender")
            fig, ax = plt.subplots()
            sns.countplot(x='sex', hue='target', data=df, ax=ax, palette=['#9d50bb', '#6e48aa'])
            ax.set_xticklabels(['Female', 'Male'])
            st.pyplot(fig)

        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
        st.pyplot(fig)

    # About page
    elif app_mode == "About":
        st.title("About CardioCare AI")

        st.markdown("""
        <div class="card">
            <h3>Heart Disease Prediction System</h3>
            <p><b>Version:</b> 1.0.0</p>
            <p>This application uses machine learning to predict the likelihood of heart disease 
            based on various medical parameters.</p>

            <h4>Model Information</h4>
            <ul>
                <li><b>Algorithm:</b> Random Forest Classifier</li>
                <li><b>Dataset:</b> Cleveland Heart Disease Dataset</li>
                <li><b>Accuracy:</b> 85% (varies based on training)</li>
            </ul>

            <h4>Disclaimer</h4>
            <p>This tool is for informational purposes only and should not replace 
            professional medical advice.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()