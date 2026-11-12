import streamlit as st
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "page_title": "Heart Disease Risk Assessment",
    "page_icon": "🫀",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

MODEL_PATHS = {
    "model": "Logistic_Regression_heart.pkl",
    "preprocessor": "preprocessor.pkl",
    "columns": "columns.pkl"
}

# Feature definitions with validation rules and display labels
FEATURES = [
    {
        "name": "age",
        "label": "Age",
        "type": "number",
        "min": 1,
        "max": 120,
        "default": 40,
        "unit": "years",
        "help": "Patient's age in years"
    },
    {
        "name": "sex",
        "label": "Sex",
        "type": "select",
        "options": [(0, "Female"), (1, "Male")],
        "default": 0,
        "help": "Biological sex"
    },
    {
        "name": "cp",
        "label": "Chest Pain Type",
        "type": "select",
        "options": [
            (0, "Typical Angina"),
            (1, "Atypical Angina"),
            (2, "Non-Anginal Pain"),
            (3, "Asymptomatic")
        ],
        "default": 0,
        "help": "Type of chest pain experienced"
    },
    {
        "name": "trtbps",
        "label": "Resting Blood Pressure",
        "type": "number",
        "min": 50,
        "max": 250,
        "default": 120,
        "unit": "mm Hg",
        "help": "Resting blood pressure measurement"
    },
    {
        "name": "chol",
        "label": "Serum Cholesterol",
        "type": "number",
        "min": 50,
        "max": 600,
        "default": 200,
        "unit": "mg/dl",
        "help": "Fasting serum cholesterol level"
    },
    {
        "name": "fbs",
        "label": "Fasting Blood Sugar",
        "type": "select",
        "options": [(0, "≤ 120 mg/dl (Normal)"), (1, "> 120 mg/dl (Elevated)")],
        "default": 0,
        "help": "Fasting blood sugar threshold"
    },
    {
        "name": "restecg",
        "label": "Resting ECG",
        "type": "select",
        "options": [
            (0, "Normal"),
            (1, "ST-T Wave Abnormality"),
            (2, "Left Ventricular Hypertrophy")
        ],
        "default": 0,
        "help": "Resting electrocardiographic results"
    },
    {
        "name": "thalachh",
        "label": "Maximum Heart Rate",
        "type": "number",
        "min": 50,
        "max": 250,
        "default": 150,
        "unit": "bpm",
        "help": "Maximum heart rate achieved during exercise"
    },
    {
        "name": "exng",
        "label": "Exercise-Induced Angina",
        "type": "select",
        "options": [(0, "No"), (1, "Yes")],
        "default": 0,
        "help": "Presence of angina during exercise testing"
    },
    {
        "name": "oldpeak",
        "label": "ST Depression (Oldpeak)",
        "type": "number",
        "min": 0.0,
        "max": 10.0,
        "default": 1.0,
        "step": 0.1,
        "unit": "mm",
        "help": "ST depression induced by exercise relative to rest"
    },
    {
        "name": "slp",
        "label": "ST Segment Slope",
        "type": "select",
        "options": [
            (0, "Upsloping"),
            (1, "Flat"),
            (2, "Downsloping")
        ],
        "default": 1,
        "help": "Slope of the peak exercise ST segment"
    },
    {
        "name": "caa",
        "label": "Major Vessels (Fluoroscopy)",
        "type": "select",
        "options": [(i, str(i)) for i in range(5)],
        "default": 0,
        "help": "Number of major vessels colored by fluoroscopy"
    },
    {
        "name": "thall",
        "label": "Thalassemia",
        "type": "select",
        "options": [
            (0, "Normal"),
            (1, "Fixed Defect"),
            (2, "Reversible Defect"),
            (3, "Other")
        ],
        "default": 0,
        "help": "Thalassemia classification"
    }
]

# Prediction labels
PREDICTION_LABELS = {
    0: {"label": "Low Risk", "description": "No significant heart disease detected", "color": "#059669"},
    1: {"label": "High Risk", "description": "Heart disease indicators present", "color": "#DC5A5A"}
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> Tuple[Any, Any, list]:
    """Load trained model, preprocessor, and feature columns."""
    try:
        model = joblib.load(MODEL_PATHS["model"])
        preprocessor = joblib.load(MODEL_PATHS["preprocessor"])
        columns = joblib.load(MODEL_PATHS["columns"])
        return model, preprocessor, columns
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()


def validate_inputs(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate all input fields meet requirements."""
    for feature in FEATURES:
        name = feature["name"]
        value = input_data.get(name)

        if value is None:
            return False, f"Missing required field: {feature['label']}"

        if feature["type"] == "number":
            try:
                val = float(value)
                if val < feature["min"] or val > feature["max"]:
                    return False, f"{feature['label']} must be between {feature['min']} and {feature['max']}"
            except (ValueError, TypeError):
                return False, f"{feature['label']} must be a valid number"

        elif feature["type"] == "select":
            valid_options = [opt[0] for opt in feature["options"]]
            if value not in valid_options:
                return False, f"{feature['label']} must be one of: {valid_options}"

    return True, ""


def render_feature_input(feature: Dict[str, Any]) -> Any:
    """Render a single feature input based on its type."""
    name = feature["name"]
    label = f"{feature['label']}"

    if feature.get("unit"):
        label += f" ({feature['unit']})"

    if feature["type"] == "number":
        return st.number_input(
            label,
            min_value=feature["min"],
            max_value=feature["max"],
            value=feature["default"],
            step=feature.get("step", 1),
            help=feature["help"],
            key=f"input_{name}"
        )

    elif feature["type"] == "select":
        options = feature["options"]
        option_values = [opt[0] for opt in options]
        option_labels = [f"{opt[0]}: {opt[1]}" for opt in options]

        selected_idx = option_values.index(feature["default"])
        selected_label = st.selectbox(
            label,
            options=option_labels,
            index=selected_idx,
            help=feature["help"],
            key=f"input_{name}"
        )
        # Extract value from label
        return option_values[option_labels.index(selected_label)]


def render_prediction_result(prediction: int, probability: float = None) -> None:
    """Render the prediction result with professional styling."""
    pred_info = PREDICTION_LABELS.get(prediction, PREDICTION_LABELS[0])

    # Main result card
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem;
            border-radius: 8px;
            background-color: {pred_info['color']}15;
            border: 2px solid {pred_info['color']};
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="
                    width: 48px;
                    height: 48px;
                    border-radius: 50%;
                    background-color: {pred_info['color']};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 1.5rem;
                    font-weight: 600;
                ">
                    {prediction}
                </div>
                <div>
                    <h3 style="margin: 0; color: {pred_info['color']}; font-size: 1.25rem;">
                        {pred_info['label']} ({prediction}: {pred_info['label']})
                    </h3>
                    <p style="margin: 0.25rem 0 0 0; color: #FFFFFF;">
                        {pred_info['description']}
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if probability is not None:
        # Probability gauge
        prob_pct = probability * 100
        bar_color = pred_info['color']

        st.markdown(
            f"""
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight: 500;">Model Confidence</span>
                    <span style="font-weight: 600; color: {bar_color};">{prob_pct:.1f}%</span>
                </div>
                <div style="
                    height: 8px;
                    background-color: #CBD5E1;
                    border-radius: 4px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {prob_pct}%;
                        height: 100%;
                        background-color: {bar_color};
                        border-radius: 4px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_input_summary(input_data: Dict[str, Any]) -> None:
    """Render a collapsible summary of all inputs."""
    with st.expander("Input Summary", expanded=False):
        col1, col2 = st.columns(2)

        features_list = [(f["name"], f["label"]) for f in FEATURES]
        mid = len(features_list) // 2

        for idx, (name, label) in enumerate(features_list):
            target_col = col1 if idx < mid else col2
            value = input_data.get(name, "N/A")

            # Find display label for select fields
            feature_def = next(f for f in FEATURES if f["name"] == name)
            if feature_def["type"] == "select":
                opt_map = {opt[0]: opt[1] for opt in feature_def["options"]}
                value = f"{value}: {opt_map.get(value, 'Unknown')}"
            elif feature_def.get("unit"):
                value = f"{value} {feature_def['unit']}"

            target_col.text(f"{label}: {value}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(**APP_CONFIG)

    # Load model artifacts
    model, preprocessor, columns = load_artifacts()

    # Professional CSS
    st.markdown("""
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom styling */
        .stApp { background-color: #252B33; color: #FFFFFF; }
        .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp span, .stApp [data-testid="stMarkdownContainer"] { color: #FFFFFF; }
        .stForm { background-color: #343B45; }
        .stNumberInput input, .stSelectbox [data-baseweb="select"] > div { background-color: #414A56; color: #FFFFFF; }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }

        h1 {
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: #FFFFFF;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        .stForm {
            background-color: #343B45;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #536B6B;
        }

        .stButton > button {
            width: 100%;
            background-color: #8B5E3C;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            border-radius: 8px;
            transition: background-color 0.2s;
        }

        .stButton > button:hover {
            background-color: #704A32;
        }

        .stButton > button:disabled {
            background-color: #6B7280;
        }

        /* Section headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #496B68;
        }

        /* Disclaimer */
        .disclaimer {
            font-size: 0.85rem;
            color: #E5E7EB;
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #75634B;
        }

        /* Number input styling */
        .stNumberInput > div > div > input {
            font-family: monospace;
        }

        /* Selectbox styling */
        .stSelectbox > div > div {
            font-family: inherit;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Clinical decision support tool using machine learning. '
        'Enter patient parameters to generate a risk prediction.</p>',
        unsafe_allow_html=True
    )

    # Initialize session state for inputs
    session_state = st.session_state
    if "input_data" not in session_state:
        session_state.input_data = {}

    # Form
    with st.form("assessment_form", clear_on_submit=False):

        # Personal Information Section
        st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)

        personal_features = [f for f in FEATURES if f["name"] in ["age", "sex"]]
        for feature in personal_features:
            session_state.input_data[feature["name"]] = render_feature_input(feature)

        # Medical Information Section
        st.markdown('<div class="section-header">Medical Information</div>', unsafe_allow_html=True)

        medical_features = [f for f in FEATURES if f["name"] not in ["age", "sex"]]

        # Group medical features into rows for better layout
        for i in range(0, len(medical_features), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j < len(medical_features):
                    feature = medical_features[i + j]
                    with col:
                        session_state.input_data[feature["name"]] = render_feature_input(feature)

        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Assess Risk", type="primary", use_container_width=True)

    # Process prediction
    if submitted:
        # Validate inputs
        is_valid, error_msg = validate_inputs(session_state.input_data)

        if not is_valid:
            st.error(f"Validation Error: {error_msg}")
        else:
            with st.spinner("Running assessment..."):
                try:
                    # Create input DataFrame
                    input_df = pd.DataFrame([session_state.input_data])
                    input_df = input_df[columns]

                    # Preprocess
                    processed_data = preprocessor.transform(input_df)

                    # Predict
                    prediction = model.predict(processed_data)[0]

                    # Get probability if available
                    probability = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(processed_data)[0]
                        probability = proba[prediction]

                    # Display results
                    st.markdown("---")
                    st.markdown("### Assessment Result")
                    render_prediction_result(prediction, probability)
                    render_input_summary(session_state.input_data)

                except Exception as e:
                    st.error(f"Assessment failed: {str(e)}")
                    st.exception(e)

    # Footer disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
            It does not constitute medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare professional for medical decisions.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()