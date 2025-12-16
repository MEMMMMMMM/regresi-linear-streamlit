import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t as tdist, f as fdist

st.set_page_config(page_title="Regresi Linear Berganda", layout="wide")
st.title("Analisis Regresi Linier Berganda")

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    cols = df.columns.tolist()

    y_var = st.selectbox("Pilih Variabel Y", cols)
    x_vars = st.multiselect("Pilih Variabel X", cols)

    alpha = st.number_input("Alpha", value=0.05)

    log_y = st.checkbox("Log(Y)")
    log_x = st.checkbox("Log(X)")

    if st.button("Jalankan Regresi"):

        data = df[[y_var] + x_vars].dropna()
        X = data[x_vars].astype(float)
        y = data[y_var].astype(float)

        if log_y:
            y = np.log(y + 1)
        if log_x:
            X = X.apply(lambda x: np.log(x + 1))

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred

        n = len(X)
        k = len(x_vars)

        sse = np.sum(residuals**2)
        ssr = np.sum((y_pred - y.mean())**2)
        mse = sse / (n - k - 1)

        r2 = model.score(X, y)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        st.subheader("Model Summary")
        st.write(f"R² : {r2:.4f}")
        st.write(f"Adjusted R² : {adj_r2:.4f}")
        st.write(f"Intercept : {model.intercept_:.4f}")

        coef_df = pd.DataFrame({
            "Variabel": x_vars,
            "Koefisien": model.coef_
        })

        st.subheader("Koefisien")
        st.dataframe(coef_df)
