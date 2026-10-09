# House Price Prediction — Streamlit ML Project

A small end-to-end machine-learning app built for `House Price Prediction Dataset(1).csv`.

## Dataset columns
`Id`, `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`, `Location`, `Condition`, `Garage`, `Price`

The app predicts `Price`; `Id` is excluded as an identifier. It uses a Random Forest regression pipeline with preprocessing for numeric and categorical columns.

## Run locally

1. Install Python 3.10 or newer.
2. Unzip this project and open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Features
- Dataset preview and missing-value overview
- Configurable test split, number of trees, and random seed
- Held-out MAE, RMSE, and R² metrics
- Actual-vs-predicted chart and price distribution
- Interactive form to estimate a house price
- Optional CSV upload (must contain a numeric `Price` column)

## Notes
Metrics reflect this dataset and split; they do not guarantee performance on real-world housing markets. For a serious deployment, validate on representative, recent data and consider location-specific features.
