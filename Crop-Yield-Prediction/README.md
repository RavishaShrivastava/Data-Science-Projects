# Project Title:
**Crop Yield Prediction Using Polynomial Regression**

# Overview:
This project applies polynomial regression to model and predict crop yield (in quintals per acre) based on various agricultural features such as fertilizer usage, nitrogen, phosphorus, potassium levels, rainfall, and temperature. It demonstrates how non-linear relationships in agricultural data can be captured using polynomial transformations.

# 📁 Dataset: crop_yield_dataset.xlsx
Columns:
    -Fertilizer
    -Nitrogen
    -Phosphorus
    -Potassium
    -Rainfall (mm)
    -Temperature
    -Yield (Q/acre) (target)

#  Machine Learning Approach:
Algorithm: Polynomial Regression (degree = 2)

# Libraries Used:
    pandas, numpy for data handling
    scikit-learn for modeling
    matplotlib for visualization

# Steps:
    Load and preprocess the data
    Transform features using PolynomialFeatures
    Train a linear regression model
    Evaluate performance using Mean Squared Error (MSE) and R² Score
    Visualize actual vs predicted yields


#  Requirements:
Install dependencies using pip:
    pip install pandas numpy scikit-learn matplotlib openpyxl

# 📝 Notes:
This model helps understand how soil and environmental features affect crop yield.
You can improve it by:
    Trying higher polynomial degrees
    Performing cross-validation
    Adding more features like soil pH, humidity, etc.

# 📊 Conclusion & Analysis – Polynomial Regression on Crop Yield Dataset

**Model:** Polynomial Regression (degree = 2)

##  Performance Metrics
- **Mean Squared Error (MSE):** 0.446  
- **R² Score (Coefficient of Determination):** 0.899

These results suggest that the model fits the data well. An R² score of 0.899 indicates that approximately 89.9% of the variability in crop yield can be explained by the input features.

## 📈 Plot Analysis – Actual vs Predicted
- The scatter plot shows predicted crop yields (Y-axis) vs. actual crop yields (X-axis).
- Most data points lie close to the 45-degree reference line, suggesting strong predictive accuracy.
- No major outliers are observed, indicating the model generalizes well on test data.

##  Interpretation
- The model captures the nonlinear relationship between features (like fertilizer usage, nutrient levels, rainfall, and temperature) and crop yield.
- Polynomial terms helped improve accuracy compared to a simple linear regression.

##  Suggestions for Improvement
- Perform k-fold cross-validation to ensure the model is not overfitting.
- Experiment with higher-degree polynomials (3 or 4), but monitor for overfitting.
- Include additional variables like soil pH, crop variety, sowing method, etc., for better generalization.
- Apply regularization (Ridge/Lasso) to handle any multicollinearity.

##  Final Note
This polynomial regression model provides a solid base for yield prediction, especially for small to mid-sized datasets with nonlinear interactions. It can be scaled or refined further based on domain knowledge and data availability.
