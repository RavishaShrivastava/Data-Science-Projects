# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_excel("crop_yield_dataset.xlsx")

# ⚠️ NOTE: make sure the spelling matches your column name exactly
feature_cols = ['Fertilizer', 'Nitrogen', 'Phosphorus',
                'Potassium', 'Rainfall (mm)', 'Temperature'] 

X = df[feature_cols]
y = df['Yield (Q/acre)']

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# --- Build model: scaling + polynomial terms + linear reg ---
poly_degree = 2
model = make_pipeline(
    StandardScaler(),                # keeps poly terms numerically stable
    PolynomialFeatures(degree=poly_degree, include_bias=False),
    LinearRegression()
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Metrics ---
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MSE  : {mse:.3f}")
print(f"R²   : {r2:.3f}")

# --- Quick sanity‑check plot: actual vs predicted ---
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linewidth=2)
plt.xlabel("Actual yield (Q/acre)")
plt.ylabel("Predicted yield (Q/acre)")
plt.title("Polynomial reg (degree=2) – actual vs predicted")
plt.tight_layout()
plt.show()
