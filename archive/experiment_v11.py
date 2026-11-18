
import joblib

from experiment_v10 import FEATURES

model = joblib.load("random_forest_model.joblib")

print("--- EXPERIMENT V11: MODEL INTEGRATION ---")
print("Model loaded successfully!")
print("Features:", FEATURES)

# Example request features
sample_request = [[
    10,     # request_count
    2.0,    # request_rate
    0.5,    # mean_interval
    0.1     # interval_std
]]

prediction = model.predict(sample_request)[0]

if prediction == 1:
    print("Prediction: ATTACK")
else:
    print("Prediction: LEGITIMATE")
