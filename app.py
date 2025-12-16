import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t as tdist, f as fdist

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Analisis Regresi",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS (Modern Professional)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
.card {
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.title {
    font-size: 26px;
    font-weight: 700;
    text-align: center;
}
.sub {
    color: #555;
    text-align: center;
    margin-bottom: 1rem;
}
.badge-good {
    color: white;
    background: #2ecc71;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 12px;
}
.badge-bad {
    color: white;
    background: #e74c3c;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='title'>Analisis Regresi Linear Berganda</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Uji t · Uji F · Model Summary</div>", unsafe_allow_html=True)

# =====================================================
# INPUT DATA
# =====================================================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    alpha = st.number_input("Alpha", value=0.05, step=0.01)

    log_y = st.checkbox("Transformasi Log(Y)")
    log_x = st.checkbox("Transformasi Log(X)")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROCESS
# =====================================================
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        y_var = st.selectbox("Variabel Dependen (Y)", df.columns)
        x_vars = st.multiselect("Variabel Independen (X)", df.columns)

        run = st.button("Jalankan Analisis")

        st.markdown("</div>", unsafe_allow_html=True)

    if run and y_var and x_vars:

        data = df[[y_var] + x_vars].dropna()
        X = data[x_vars].astype(float)
        y = data[y_var].astype(float)

        if log_y:
            y = np.log(y + 1)
        if log_x:
            X = X.apply(lambda c: np.log(c + 1))

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred

        n = len(X)
        k = len(x_vars)

        # ======================
        # STATISTIK
        # ======================
        sse = np.sum(residuals ** 2)
        ssr = np.sum((y_pred - y.mean()) ** 2)
        mse = sse / (n - k - 1)

        Xmat = np.column_stack([np.ones(n), X])
        cov = mse * np.linalg.inv(Xmat.T @ Xmat)
        se = np.sqrt(np.diag(cov))

        t_vals = model.coef_ / se[1:]
        p_vals = 2 * (1 - tdist.cdf(np.abs(t_vals), df=n - k - 1))

        r2 = model.score(X, y)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        f_stat = (ssr / k) / mse
        f_pval = 1 - fdist.cdf(f_stat, k, n - k - 1)

        # =====================================================
        # MODEL SUMMARY
        # =====================================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Summary")
        st.write(f"**R²** : {r2:.4f}")
        st.write(f"**Adjusted R²** : {adj_r2:.4f}")
        st.write(f"**F-statistic** : {f_stat:.4f}")
        st.write(f"**F p-value** : {f_pval:.4f}")
        st.write(f"**n** : {n}")
        st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # UJI T PER VARIABEL
        # =====================================================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Uji t Variabel")

        for var, beta, tval, pval in zip(x_vars, model.coef_, t_vals, p_vals):

            badge = (
                "<span class='badge-good'>Signifikan</span>"
                if pval < alpha else
                "<span class='badge-bad'>Tidak Signifikan</span>"
            )

            st.markdown(
                f"""
                **{var}**  
                β = {beta:.4f}  
                t = {tval:.4f}  
                p = {pval:.4f} {badge}
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
