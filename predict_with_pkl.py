import joblib
import pandas as pd

# Run from the project folder. This loads the model trained on the supplied dataset.
bundle = joblib.load("house_price_model.pkl")
model = bundle["model"]
features = bundle["features"]

sample = pd.DataFrame([{
    feature: 0 for feature in features
}])
# Replace sample values with realistic values for your dataset before predicting.
prediction = model.predict(sample[features])[0]
print(f"Predicted house price: {prediction:,.2f}")
