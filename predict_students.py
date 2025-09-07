import joblib
import pandas as pd

#load trained model
model = joblib.load("student_risk_model.pkl")

# New student details
new_student = pd.DataFrame([{
    "attendance": 100,
    "test_score_avg": 40,
    "attempts": 2,
    "fee_paid": 0
}])

prediction = model.predict(new_student)
# print("Predicted Risk status ",prediction[0]) 
if prediction[0] == 1:
    print("⚠️ Student is AT RISK")
else:
    print("✅ Student is NOT at risk")
