# 📉 Telecom Customer Churn Prediction

A machine learning project to predict which telecom customers are likely to churn, enabling telecom companies to identify at-risk customers and take proactive retention strategies.

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs significantly more than retaining an existing one. This project builds a classification model to identify customers likely to churn before they leave.

---

## 📂 Project Structure

```text
Telecom-Churn-Prediction/
├── data/
│   ├── raw_telecom.csv
│   └── cleaned_telecom.csv
│
├── notebook/
│   ├── cleaning_eda.ipynb
│   └── model_building.ipynb
│
├── models/
│   └── churn_model_final.pkl
│
├── .gitignore
├── LICENSE
└── README.md
|__requirements.txt
```

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Target Variable:** Churn (Yes/No)
- **Features:** Customer demographics, services, billing information, and account details

---

## 🔧 Workflow

### 1. Data Cleaning
- Converted `TotalCharges` to numeric
- Handled missing values
- Removed duplicate records
- Dropped irrelevant columns (`customerID`)

### 2. Exploratory Data Analysis
- Performed univariate, bivariate, and multivariate analysis
- Analyzed churn patterns among customer groups

Key findings:

- Customers with month-to-month contracts showed higher churn
- Customers without online security and tech support were more likely to churn
- Short-tenure customers had higher churn probability

### 3. Feature Engineering

| Feature | Description |
|---|---|
| charge_ratio | MonthlyCharges / TotalCharges |
| tenure_group | Customers grouped by tenure |
| service_count | Number of subscribed services |
| avg_monthly_revenue | Historical average spending |
| is_new_customer | Tenure ≤ 6 months |
| high_value_at_risk | High-paying customers with low tenure |

---

### 4. Preprocessing

- Label Encoding for binary categorical features
- One-Hot Encoding for multi-class features
- Applied SMOTE to balance classes
- Standardized numerical features using StandardScaler

---

### 5. Model Training

Trained multiple classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

### 6. Hyperparameter Tuning

Applied:

- RandomizedSearchCV
- 5-fold Cross Validation
- ROC-AUC optimization

---

### 7. Threshold Optimization

Instead of using the default threshold (0.50), multiple thresholds were evaluated.

Selected threshold:

```text
Threshold = 0.35
```

This improved churn recall and overall model performance.

---

## 📈 Results

### Model Comparison

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 79.89% | 85.89% |
| Random Forest | 79.08% | 85.14% |
| XGBoost | 78.99% | 84.10% |
| Decision Tree | 76.85% | 81.10% |

---

## 🏆 Final Selected Model — Tuned Logistic Regression

| Metric | Score |
|---|---:|
| ROC AUC | 0.8587 |
| Recall (Churn) | 0.76 |
| Precision (Churn) | 0.58 |
| F1 Score | 0.66 |
| Accuracy | 78% |

Confusion Matrix:

```text
[[1217 322]
 [137 437]]
```

> ROC-AUC score of **0.8587** indicates strong class separation ability.

> Recall of **0.76** means the model successfully identifies **76% of customers who actually churn**.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | ML algorithms and preprocessing |
| XGBoost | Boosting algorithm |
| Imbalanced-learn | SMOTE implementation |
| Pickle | Model saving |

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/agilan-0110/Telecom-Churn-Prediction.git

# Move into project folder
cd Telecom-Churn-Prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Start Jupyter Notebook
jupyter notebook
```

Run notebooks in order:

1. `cleaning_eda.ipynb`
2. `model_building.ipynb`

---

## 🔮 Future Improvements

- Build Streamlit web application
- Deploy model on cloud
- Add real-time churn prediction
- Experiment with advanced ensemble models

---

## 👤 Author

**Agilan T**  
AI/ML Enthusiast | AI & Data Science Student

GitHub: https://github.com/agilan-0110

---
