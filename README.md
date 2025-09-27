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
# Google Stock Price Prediction & Analysis using Machine Learning 📈🤖


---

## 🎯 Research Objectives
1. To evaluate the effectiveness of multiple ML algorithms in predicting Google stock prices.  
2. To compare regression vs classification approaches for stock market forecasting.  
3. To identify the most reliable model for short-term stock price movement prediction.  
4. To apply clustering techniques for exploratory market behavior grouping.  

---

## 📊 Methodology
1. **Dataset**: Daily Google stock prices (`open`, `high`, `low`, `close`, `volume`).  
   - Target variable for regression: **Next-day closing price**.  
   - Target variable for classification: **Price movement (up/down)**.  

2. **Preprocessing**:  
   - Sorted data by date  
   - Created `next_day_close` and binary `target` label  
   - Train-test split (80-20 ratio)  
   - Label encoding for classification targets  

3. **Algorithms Applied**:
   - **Regression**: Linear Regression, Lasso Regression  
   - **Classification**: Logistic Regression, KNN, Gaussian Naive Bayes, Bernoulli Naive Bayes, Decision Tree, Random Forest, SVM  
   - **Clustering**: K-Means (with Elbow method to find optimal clusters)  

4. **Evaluation Metrics**:  
   - Regression: R² Score, Mean Squared Error (MSE), Mean Absolute Error (MAE)  
   - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC  

---

## 📈 Results & Analysis
- **Random Forest Classifier**: Highest accuracy (~82.1%) and strong precision (85%).  
- **Decision Tree & Gaussian Naive Bayes**: Moderate performance but less robust.  
- **Logistic Regression, Bernoulli NB, SVM**: Similar performance, limited differentiation.  
- **KNN**: Lowest accuracy among classifiers.  
- **Regression models**: Poor R² scores → unsuitable for stock movement prediction.  
- **K-Means Clustering**: Demonstrated clear grouping of stock price behaviors.  

---

## 📌 Conclusion
The study finds that **ensemble methods (Random Forest)** outperform traditional ML models in short-term stock prediction. Regression models were not effective, highlighting the importance of classification-based approaches. Future improvements can include advanced deep learning models (LSTM/GRU/Transformers) and sentiment analysis integration for enhanced predictive power.  

---

## 🔮 Future Work
- Incorporating **time-series deep learning models** (LSTMs, GRUs, Transformers).  
- Adding **sentiment analysis** from financial news and social media.  
- Hyperparameter tuning for further performance gains.  
- Testing on **different stock datasets** for generalization.  


