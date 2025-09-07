import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Create dummy data
data = {
    "Attendance": [60, 85, 72, 90, 55, 40, 95, 68, 82, 77],
    "TestScore": [45, 70, 65, 80, 50, 30, 88, 60, 73, 69],
    "AttemptsExhausted": [2, 0, 1, 0, 3, 4, 0, 2, 1, 1],
    "FeeDelayDays": [10, 0, 5, 0, 20, 30, 0, 7, 2, 0],
    "RiskLabel": ["High", "Low", "Medium", "Low", "High", "High", "Low", "Medium", "Low", "Medium"]
}

df = pd.DataFrame(data)

# Step 2: Encode labels (High/Medium/Low → numbers)
df["RiskLabel"] = df["RiskLabel"].map({"Low": 0, "Medium": 1, "High": 2})

# Step 3: Features (X) and Target (y)
X = df[["Attendance", "TestScore", "AttemptsExhausted", "FeeDelayDays"]]
y = df["RiskLabel"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Feature importance
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.title("Feature Importance")
plt.show()
