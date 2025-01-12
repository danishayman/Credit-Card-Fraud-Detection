# Credit Card Fraud Detection 💳 

## Overview 🔎
This project implements various machine learning models to detect fraudulent credit card transactions. The system uses a dataset of credit card transactions to train and evaluate different classification algorithms, helping to identify potentially fraudulent activities.

## Dataset 📊
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders.

Dataset characteristics:
- 284,807 transactions
- 492 frauds (0.172% of all transactions)
- 31 features (28 principal components + Time + Amount + Class)
- Highly imbalanced dataset

## Features 🌟
- Comprehensive Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Implementation of multiple machine learning models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - Bagging Classifier
  - Extra Trees Classifier
  - Stochastic Gradient Descent Classifier
  - Voting Classifier
- Handling imbalanced data using:
  - Undersampling
  - Oversampling (SMOTE)
- Model evaluation and comparison
- Model persistence for future use

## Technologies Used 🛠️
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Imbalanced-learn (SMOTE)
- Joblib

## Key Findings 📈
- Best performing models after SMOTE:
  - Decision Tree Classifier (99.82% accuracy)
  - Logistic Regression (94.52% accuracy)
- Successfully handled the class imbalance problem
- Time of transaction doesn't significantly impact fraud detection
- Fraudulent transactions typically involve smaller amounts

## Getting Started 🚀

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn joblib
```

### Usage
1. Clone the repository
2. Download the dataset from Kaggle
3. Run the Jupyter notebook
4. Use the saved model for predictions

## Model Performance 📊

### After SMOTE:
- Decision Tree Classifier:
  - Accuracy: 99.82%
  - Precision: 99.74%
  - Recall: 99.89%
  - F1 Score: 99.82%

## Future Improvements 🔮
1. Feature engineering to create new relevant features
2. Implement deep learning models
3. Real-time transaction monitoring system
4. Add more evaluation metrics
5. Create a web interface for predictions

## Contributing 🤝
Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.


## Acknowledgments 🙏
- Dataset provided by Kaggle and the Machine Learning Group of ULB (Université Libre de Bruxelles)
- Thanks to all contributors and the open-source community

## Contact 📧
Feel free to reach out for any questions or suggestions!

---