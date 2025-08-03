# 📰 Fake News Detection with Machine Learning

Welcome! This project is a simple yet powerful attempt to solve a very real problem: identifying whether a news article is **True** or **fake** using machine learning.

---

## 📁 Dataset Overview

⚠️ Note: Dataset is not included due to size. 
Please upload True.csv and Fake.csv manually or modify the code to load from your local/Google Drive.

- `True.csv`: [Dataset link of **True.csv**](https://docs.google.com/spreadsheets/d/1oaCanPcX9ZIHmiOZonWmCORqbwR3RKC5goa8BRjMGak/edit?usp=drive_link)
- `Fake.csv`: [Dataset link of **fake.csv**](https://docs.google.com/spreadsheets/d/19zkYZaciqHrabm_vTj9hGfUjlFwIvlJC8XXF5qga7cE/edit?usp=drive_link)

Each entry includes a news article's full text, along with other metadata like title, subject, and date (which we remove during preprocessing).

---

## Technologies Used
Python
Pandas
Scikit-learn
TfidfVectorizer
Regular Expressions (for text cleaning)
Joblib (for saving the trained model)

---
## 🚀 Project Highlights

1. **Data Preprocessing**
   - Combined real and fake datasets
   - Cleaned the text (removed digits, special characters, URLs, etc.)
   - Dropped irrelevant columns like `title`, `subject`, and `date`
   - Shuffled and reset the data for better generalization

2. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to convert text into numerical vectors

3. **Model Building**
   - Trained four different machine learning models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting

4. **Evaluation**
   - Measured training and testing accuracy
   - Created a `manual_testing()` function to allow real-time input and prediction

---

## 📊 Model Performance

| Model                | Training Accuracy | Testing Accuracy |
|---------------------|-------------------|------------------|
| Logistic Regression | ~99%              | ~98%             |
| Decision Tree       | ~100%             | ~92%             |
| Random Forest       | ~100%             | ~96%             |
| Gradient Boosting   | ~98%              | ~96%             |

*Note: These numbers may vary slightly based on data shuffling.*

---
##  Future Work
Here are some ideas to take this project to the next level:

-- Streamlit Web App: Build and deploy an interactive web interface to allow users to test news live

-- Ensemble Voting Classifier: Combine multiple models for more reliable predictions

-- Deep Learning Models: Add LSTM, BiLSTM, or transformer-based models like BERT for context-aware classification

-- Multilingual Support: Extend the model to support fake news detection in other languages

-- Dashboard for Analytics: Display predictions, model confidence scores, and live feedback in a visual dashboard

---
## 💬 Try It Yourself

Use the built-in function to test your own news:

```python
manual_testing("Breaking news: Government introduces new education policy.")

