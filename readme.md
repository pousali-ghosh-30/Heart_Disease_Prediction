# Heart Disease Prediction

A machine learning project that predicts whether a patient is likely to have heart disease based on clinical attributes such as age, blood pressure, cholesterol, chest pain type, and ECG results.

The project includes both a training workflow and a Streamlit web app for interactive prediction.

## Overview

This application uses a trained Logistic Regression model to assess heart disease risk from medical input values. The model is saved and loaded with joblib, and the app uses a preprocessing pipeline to prepare incoming data before prediction.

## Features

- Predicts heart disease risk using a trained ML model
- Interactive web interface built with Streamlit
- Input validation for all medical parameters
- Preprocessing pipeline for model input consistency
- Saved model artifacts for reuse without retraining

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- joblib

## Dataset

The project uses the Heart Attack Analysis & Prediction Dataset.

Source: https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset

The dataset contains 13 clinical features and a target variable indicating whether heart disease is present.

### Key Features

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of ST segment
- Number of major vessels
- Thalassemia

## Project Structure

```text
Heart_Disease_Prediction/
├── app.py
├── Logistic_Regression_heart.pkl
├── preprocessor.pkl
├── columns.pkl
├── requirements.txt
├── readme.md
└── venv/
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pousali-ghosh-30/Heart_Disease_Prediction.git
```

2. Navigate to the project folder:

```bash
cd Heart_Disease_Prediction
```

3. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

4. Activate the virtual environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

Fill in the patient details in the form and click the prediction button to view the risk result.

## Model Pipeline

The project includes:

- Data preprocessing
- Train/test split
- Feature scaling and transformation
- Model training
- Evaluation using performance metrics
- Model serialization for later inference

## Model Details

The application currently uses a saved Logistic Regression model trained on the heart disease dataset. The trained model and preprocessing objects are stored in the project root as pickle files.

## Example Prediction Output

The app classifies patients into:

- Low Risk
- High Risk

The output includes a prediction label and a confidence-style probability measure.

## Disclaimer

This project is intended for educational and research purposes only. It should not be considered a professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.

## License

This project is distributed for learning and demonstration purposes.

