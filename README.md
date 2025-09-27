# Google Stock Price Prediction & Analysis 📈

This project applies multiple **machine learning algorithms** to analyze and predict **Google's stock price movements**.  
It includes both **regression models** (for predicting next-day closing prices) and **classification models** (for predicting stock movement: up/down).  

---

## 🚀 Features
- Preprocessing and feature engineering of Google stock dataset  
- Implementation of multiple machine learning models:
  - **Regression Models**: Linear Regression, Lasso Regression  
  - **Classification Models**: Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes (Gaussian & Bernoulli), Decision Tree, Random Forest, Support Vector Machine (SVM)  
- **Performance Evaluation**:
  - Accuracy, R² Score, Precision, Recall, F1-Score  
  - Confusion Matrix & Classification Reports  
- **Visualization**:
  - Actual vs Predicted stock prices  
  - ROC Curves for classifiers  
  - K-Means clustering with Elbow method  

---

## 📊 Dataset
- The dataset used is **Google daily stock prices** (`googl_daily_prices.csv`).  
- Key columns: `open`, `high`, `low`, `close`, `volume`, and a derived target `up/down`.  

