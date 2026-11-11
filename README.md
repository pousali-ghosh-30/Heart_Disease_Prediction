# ❤️ Heart Disease Prediction Using Machine Learning

## Project Overview

This project aims to predict the presence of heart disease using Machine Learning algorithms. It analyzes various medical attributes such as age, cholesterol, blood pressure, chest pain type, and maximum heart rate to classify the input data.

The project uses Python and Scikit-learn for machine learning and Streamlit to develop an interactive web application.

## Objectives

- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Apply different Machine Learning classification algorithms.
- Evaluate and compare model performance.
- Save the trained model for future predictions.
- Develop an interactive frontend using Streamlit.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Frontend:** Streamlit
- **Development Environment:** Kaggle Notebook, VS Code

## Dataset

**Dataset:** Heart Attack Analysis & Prediction Dataset

**Source:** (https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset)

The dataset contains 13 input features and one target variable (`output`).

### Features Used

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Sex |
| cp | Chest pain type |
| trtbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalachh | Maximum heart rate |
| exng | Exercise-induced angina |
| oldpeak | ST depression |
| slp | Slope |
| caa | Number of major vessels |
| thall | Thalassemia category |
| output | Target variable |

## 🔍 Project Workflow

1. **Data Collection:** Downloaded the dataset from Kaggle.
2. **Exploratory Data Analysis:** Examined data distributions, missing values, duplicates, and feature relationships.
3. **Data Cleaning:** Checked and removed duplicate records.
4. **Data Preprocessing:** Separated features and target, performed train-test splitting, scaling, and categorical encoding.
5. **Model Training:** Trained multiple classification algorithms.
6. **Model Evaluation:** Compared models using Accuracy and F1 Score.
7. **Model Saving:** Saved the trained model and preprocessing objects using Joblib.
8. **Frontend Development:** Built an interactive application using Streamlit.

## 🤖 Machine Learning Algorithms

The following algorithms were implemented and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

The Streamlit application currently uses the saved Logistic Regression model.

## Model Evaluation

The models are evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Classification Report

## Project Structure

```text
Heart_Disease_Prediction/
│
├── app.py
├── Logistic_Regression_heart.pkl
├── preprocessor.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pousali-ghosh-30/Heart-Disease-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Heart-Disease-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## Application Features

- Interactive user input form.
- Accepts 13 medical input features.
- Automatically preprocesses user inputs.
- Uses the trained Logistic Regression model.
- Displays the predicted classification result.

## Future Enhancements

- Add model selection functionality.
- Improve the user interface.
- Display model performance comparisons.
- Add interactive data visualizations.
- Deploy the application online.

## Disclaimer

This project is developed for educational and academic purposes only. The predictions are based on a dataset classification task and must not be considered a medical diagnosis or a prediction of an individual's future heart attack. Always consult a qualified healthcare professional for medical advice.


---

⭐ If you find this project interesting, consider giving the repository a star!
