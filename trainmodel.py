import pandas as pd  # for handling CSV / tabular data
from sklearn.model_selection import train_test_split  # for splitting dataset
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # performance check
import joblib  # for saving and loading trained model

# Load dataset
df = pd.read_csv("student_data.csv")
print(df.head())

# Features (input for the model)
X = df.drop(columns=["risk_status", "student_id"])

# Target (what we want to predict)
y = df["risk_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# test_size=0.2 → 20% of data goes to testing.
# random_state=42 → ensures reproducibility.

# Initialize model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
                # n_estimators=100 → number of trees in the forest.
# Train model
rf.fit(X_train, y_train)
                #fit → trains the model using training data.

# Predict on test set
y_pred = rf.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importances:\n", importances.sort_values(ascending=False))

joblib.dump(rf, "student_risk_model.pkl")
print("Model saved as student_risk_model.pkl")



