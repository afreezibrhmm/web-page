import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("car_price_prediction_.csv")

# ---------------------------
# 2. Select Features and Target
# ---------------------------
X = df.drop(columns=["Price", "Car ID"])
y = df["Price"]

# ---------------------------
# 3. Find Categorical Columns
# ---------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ---------------------------
# 4. Preprocessing (One-Hot Encode categoricals)
# ---------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------
# 5. Create Pipeline (Preprocessing + Model)
# ---------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ---------------------------
# 6. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 7. Train the Model
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# 8. Evaluate the Model
# ---------------------------
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("Model R² Score:", score)

# ---------------------------
# 9. Save the Model
# ---------------------------
joblib.dump(model, "car_price_model.pkl")
print("Model saved as car_price_model.pkl")
