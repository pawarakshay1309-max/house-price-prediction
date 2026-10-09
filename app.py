import os
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt 

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "house_price_dataset.csv")
TARGET = "Price"

@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)

@st.cache_resource
def train_model(data, test_size, n_estimators, random_state):
    df = data.copy()
    df = df.dropna(subset=[TARGET])
    # IDs are identifiers rather than meaningful property features.
    features = [c for c in df.columns if c not in [TARGET, "Id"]]
    X, y = df[features], df[TARGET]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R²": r2_score(y_test, pred),
    }
    return model, features, numeric, categorical, metrics, X_test, y_test, pred

st.title("🏡 House Price Prediction")
st.caption("A machine-learning regression app built for the supplied house-price dataset.")
with st.sidebar:
    st.header("Model settings")
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    n_estimators = st.slider("Number of trees", 50, 500, 200, 50)
    random_state = st.number_input("Random seed", min_value=0, max_value=10000, value=42)
    st.divider()
    uploaded = st.file_uploader("Optional: upload another CSV", type=["csv"])
    st.caption("The CSV must contain a numeric `Price` target column.")

try:
    if uploaded is not None:
        data = load_csv(uploaded)
        source_name = uploaded.name
    else:
        data = load_csv(DATA_PATH)
        source_name = "bundled dataset"
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

if TARGET not in data.columns:
    st.error("The dataset must include a `Price` column.")
    st.stop()

st.markdown(f"**Data source:** {source_name}  ·  **Rows:** {len(data):,}  ·  **Columns:** {len(data.columns)}")
with st.expander("Preview dataset"):
    st.dataframe(data.head(30), use_container_width=True)
    st.write("Column types")
    st.dataframe(pd.DataFrame({"Column": data.columns, "Type": data.dtypes.astype(str).values}), use_container_width=True)
    st.write("Missing values")
    st.dataframe(data.isna().sum().rename("Missing values").to_frame(), use_container_width=True)

if len(data) < 10:
    st.warning("Please use at least 10 rows to train and evaluate the model.")
    st.stop()

try:
    model, features, numeric, categorical, metrics, X_test, y_test, pred = train_model(
        data, test_size, int(n_estimators), int(random_state)
    )
except Exception as e:
    st.error(f"Model training failed. Check that `Price` is numeric and features are usable. Details: {e}")
    st.stop()

st.subheader("Model performance")
m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error", f"{metrics['MAE']:,.0f}")
m2.metric("Root Mean Squared Error", f"{metrics['RMSE']:,.0f}")
m3.metric("R² score", f"{metrics['R²']:.3f}")
st.caption("Metrics are calculated on a held-out test set. Lower MAE/RMSE is better; R² closer to 1 is better.")

left, right = st.columns(2)
with left:
    st.subheader("Actual vs predicted")
    chart_df = pd.DataFrame({"Actual price": y_test.to_numpy(), "Predicted price": pred})
    st.scatter_chart(chart_df, x="Actual price", y="Predicted price")
with right:
    st.subheader("Price distribution")
    st.bar_chart(data[TARGET].dropna().value_counts(bins=20, sort=False).sort_index())

st.divider()
st.subheader("Predict a house price")
st.write("Enter property details below. The app retrains the model when settings or the dataset change.")
with st.form("prediction_form"):
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(features):
        container = cols[i % 3]
        series = data[col]
        if col in numeric:
            vals = pd.to_numeric(series, errors="coerce").dropna()
            default = float(vals.median()) if not vals.empty else 0.0
            low = float(vals.min()) if not vals.empty else 0.0
            high = float(vals.max()) if not vals.empty else max(default + 1, 1.0)
            if low == high:
                high = low + 1
            step = 1.0 if pd.api.types.is_integer_dtype(series.dtype) else 0.1
            inputs[col] = container.number_input(
                col, min_value=low, max_value=high, value=min(max(default, low), high), step=step
            )
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            inputs[col] = container.selectbox(col, options)
    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    row = pd.DataFrame([inputs], columns=features)
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated house price: {prediction:,.2f}")
    st.caption("This is a model estimate, not a guaranteed market valuation.")

with st.expander("About this project"):
    st.markdown("""
    - **Problem type:** supervised regression
    - **Target:** `Price`
    - **Algorithm:** Random Forest Regressor
    - **Preprocessing:** median imputation for numeric features, most-frequent imputation and one-hot encoding for categorical features
    - **Excluded field:** `Id`, because it is an identifier rather than a property characteristic
    - **Evaluation:** train/test split with MAE, RMSE and R²
    """)
