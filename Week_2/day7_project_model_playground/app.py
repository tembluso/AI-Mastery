import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Model Playground — California Housing", page_icon="🏡", layout="wide")
st.title("🏡 Model Playground — California Housing")
st.caption("Compare Linear Regression, Random Forest, and Gradient Boosting. Tune hyperparameters, inspect importances, make predictions — and visualize learning curves to diagnose under/overfitting.")

# ---------------------------
# Data loading (cached)
# ---------------------------
@st.cache_resource
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y, X.columns.tolist()

X, y, FEATURE_NAMES = load_data()
NUM_COLS = FEATURE_NAMES

# Preprocessor for Linear Regression (trees will use passthrough)
NUM_PREPROCESS = Pipeline([("scaler", StandardScaler())])
PREPROCESSOR = ColumnTransformer([("num", NUM_PREPROCESS, NUM_COLS)], remainder="drop")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Model & Hyperparameters")

MODEL_NAME = st.sidebar.selectbox("Choose model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

if MODEL_NAME == "Linear Regression":
    params = {}
elif MODEL_NAME == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 500, 200, 10)
    max_depth_opt = st.sidebar.slider("max_depth (None = 0)", 0, 30, 0, 1)
    max_depth = None if max_depth_opt == 0 else max_depth_opt
    max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", None], index=0)
    params = dict(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1)
else:
    n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 200, 50)
    learning_rate = st.sidebar.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    max_depth = st.sidebar.slider("max_depth", 1, 8, 3, 1)
    subsample = st.sidebar.select_slider("subsample", options=[0.5, 0.7, 0.8, 1.0], value=1.0)
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)

# ---------------------------
# Helpers
# ---------------------------
def build_pipeline(model_name: str, params: dict) -> Pipeline:
    """Create a preprocessing+model pipeline. Linear gets scaling; trees passthrough."""
    if model_name == "Linear Regression":
        model = LinearRegression()
        return Pipeline([("prep", PREPROCESSOR), ("model", model)])
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=random_state, **params)
        passthrough = ColumnTransformer([("num", "passthrough", NUM_COLS)], remainder="drop")
        return Pipeline([("prep", passthrough), ("model", model)])
    else:
        model = GradientBoostingRegressor(random_state=random_state, **params)
        passthrough = ColumnTransformer([("num", "passthrough", NUM_COLS)], remainder="drop")
        return Pipeline([("prep", passthrough), ("model", model)])

def train_eval(pipe: Pipeline, X, y, test_size=0.2, random_state=42):
    """Single train/test split; return fitted pipe, (R2, RMSE), and raw arrays."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    return pipe, (r2, rmse), (X_test, y_test, pred)

def cache_key(model_name: str, params: dict, test_size: float, random_state: int):
    return (model_name, tuple(sorted(params.items())), float(test_size), int(random_state))

if "trained_cache" not in st.session_state:
    st.session_state.trained_cache = {}

def get_trained_result(model_name: str, params: dict, test_size: float, random_state: int):
    key = cache_key(model_name, params, test_size, random_state)
    if key not in st.session_state.trained_cache:
        with st.spinner("Training model…"):
            pipe = build_pipeline(model_name, params)
            st.session_state.trained_cache[key] = train_eval(
                pipe, X, y, test_size=test_size, random_state=random_state
            )
    return st.session_state.trained_cache[key]

@st.cache_data
def compare_models_table(test_size: float, random_state: int) -> pd.DataFrame:
    """Side-by-side baseline comparison using fixed hyperparams."""
    configs = {
        "Linear Regression": {},
        "Random Forest": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "n_jobs": -1},
        "Gradient Boosting": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0},
    }
    rows = []
    for name, p in configs.items():
        pipe = build_pipeline(name, p)
        _, (r2_i, rmse_i), _ = train_eval(pipe, X, y, test_size=test_size, random_state=random_state)
        rows.append([name, r2_i, rmse_i])
    return pd.DataFrame(rows, columns=["Model", "R2", "RMSE"]).sort_values("R2", ascending=False)

# ---------------------------
# Learning Curves (no k-fold): grow training size, fixed validation set
# ---------------------------
@st.cache_data
def learning_curve_no_cv(model_name: str, params: dict, test_size: float, random_state: int,
                         n_points: int = 8, min_frac: float = 0.1, max_frac: float = 1.0):
    """
    Compute learning curves without cross-validation.
    - Single train/validation split controlled by test_size & random_state.
    - Shuffle training set deterministically, then fit on increasing fractions.
    Returns: train_sizes (int), train_r2 (float), val_r2 (float)
    """
    # Base split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    n_train = len(X_train)
    rng = np.random.RandomState(random_state)
    order = rng.permutation(n_train)
    X_train = X_train.iloc[order]
    y_train = y_train.iloc[order]

    # sizes to evaluate
    fracs = np.linspace(min_frac, max_frac, n_points)
    sizes = np.unique((fracs * n_train).astype(int))
    sizes = sizes[sizes >= 2]  # need at least 2 samples to fit

    train_scores = []
    val_scores = []

    for m in sizes:
        pipe = build_pipeline(model_name, params)
        pipe.fit(X_train.iloc[:m], y_train.iloc[:m])
        # R² on the m-sample training subset
        y_tr_pred = pipe.predict(X_train.iloc[:m])
        train_scores.append(r2_score(y_train.iloc[:m], y_tr_pred))
        # R² on the fixed validation set
        y_val_pred = pipe.predict(X_val)
        val_scores.append(r2_score(y_val, y_val_pred))

    return sizes.tolist(), np.array(train_scores).astype(float), np.array(val_scores).astype(float)

def plot_learning_curves(sizes, train_r2, val_r2):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, train_r2, marker="o", label="Train R²")
    ax.plot(sizes, val_r2, marker="o", label="Validation R²")
    ax.set_xlabel("Training set size (# samples)")
    ax.set_ylabel("R² score")
    ax.set_title("Learning Curves (single split)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

# ---------------------------
# PRECOMPUTE EVERYTHING (no work during rendering)
# ---------------------------
pipe, (r2, rmse), (X_test, y_test, y_pred) = get_trained_result(MODEL_NAME, params, test_size, random_state)
compare_df = compare_models_table(test_size, random_state)

# ---------------------------
# Segmented control (renders only one section)
# ---------------------------
choice = st.segmented_control(
    "Sections",
    ["Train & Inspect", "Compare Models", "Predict", "Learning Curves"],
    default="Train & Inspect"
)

if choice == "Train & Inspect":
    st.subheader("Train & Inspect current model")
    c1, c2 = st.columns(2)
    with c1: st.metric("R² (validation)", f"{r2:.3f}")
    with c2: st.metric("RMSE (validation)", f"{rmse:.3f}")

    st.markdown("### Feature importances / coefficients")
    model = pipe.named_steps["model"]

    if MODEL_NAME == "Linear Regression":
        coefs = model.coef_.ravel()
        coef_df = pd.DataFrame({
            "feature": NUM_COLS,
            "importance": np.abs(coefs),
            "signed_coef": coefs
        }).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(coef_df["feature"], coef_df["importance"])
        ax.set_xlabel("Absolute standardized coefficient")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.dataframe(coef_df.reset_index(drop=True), width="stretch")
        st.caption("Linear regression shows *direction* via the sign of the coefficient (see `signed_coef`). Magnitude indicates strength (after standardization).")
    else:
        try:
            imp_df = pd.DataFrame({
                "feature": NUM_COLS,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.set_xlabel("Feature importance")
            ax.invert_yaxis()
            st.pyplot(fig)
            st.dataframe(imp_df.reset_index(drop=True), width="stretch")
            st.caption("Tree-based models expose impurity-based feature importances.")
        except Exception as e:
            st.error(f"Could not compute feature importances: {e}")

    with st.expander("Save / load model"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save current pipeline to disk"):
                joblib.dump(pipe, "housing_pipeline.joblib")
                st.success("Saved as housing_pipeline.joblib")
        with c2:
            if st.button("📂 Load pipeline from disk"):
                try:
                    loaded = joblib.load("housing_pipeline.joblib")
                    st.success("Loaded housing_pipeline.joblib")
                except Exception as e:
                    st.error(str(e))

elif choice == "Compare Models":
    st.subheader("Side-by-side comparison (fixed settings)")
    st.dataframe(compare_df, width="stretch")
    st.caption("On this dataset you should typically see Gradient Boosting ≳ Random Forest ≫ Linear Regression.")

elif choice == "Predict":
    st.subheader("Predict with the current model")
    st.caption("Enter custom inputs. Values are in the same units as the original dataset.")
    cols = st.columns(4)
    defaults = X.median()
    inputs = {}
    for i, feat in enumerate(NUM_COLS):
        with cols[i % 4]:
            inputs[feat] = st.number_input(feat, value=float(defaults[feat]))
    if st.button("Predict house value"):
        X_new = pd.DataFrame([inputs])
        y_new = pipe.predict(X_new)[0]
        st.subheader(f"💰 Predicted median house value: **${y_new*100000:,.0f}**")
        st.caption("Note: Target is in 100k USD units in the original dataset. The displayed value multiplies by 100,000 for readability.")

else:  # Learning Curves
    st.subheader("Learning Curves")
    st.caption("We train the same pipeline on increasing training sizes (single split) and plot R² on both training and validation sets to diagnose under/overfitting.")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_points = st.number_input("Number of points", min_value=5, max_value=20, value=8, step=1)
    with c2:
        min_frac = st.slider("Min train fraction", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    with c3:
        max_frac = st.slider("Max train fraction", min_value=0.6, max_value=1.0, value=1.0, step=0.05)

    with st.spinner("Computing learning curves…"):
        sizes, train_r2, val_r2 = learning_curve_no_cv(
            MODEL_NAME, params, test_size, random_state,
            n_points=int(n_points), min_frac=float(min_frac), max_frac=float(max_frac)
        )
    fig = plot_learning_curves(sizes, train_r2, val_r2)
    st.pyplot(fig)

    # Small table for reference
    lc_df = pd.DataFrame({"train_size": sizes, "train_R2": train_r2, "val_R2": val_r2})
    st.dataframe(lc_df, width="stretch")
    st.caption(
        "Reading the chart: large gap with low validation R² ⇒ overfitting. "
        "Both low and close ⇒ underfitting. Both high and close ⇒ good generalization; more data may still help if curves are rising."
    )
