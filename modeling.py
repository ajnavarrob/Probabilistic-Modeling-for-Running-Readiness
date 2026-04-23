import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ============================================================
# 1. LOAD FILES
# ============================================================

HISTORY_FILE = "garmin_history_raw.csv"
FUTURE_FILE = "garmin_future_raw.csv"
METADATA_FILE = "garmin_model_metadata.json"

history_df = pd.read_csv(HISTORY_FILE)
future_raw_df = pd.read_csv(FUTURE_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

ACUTE_WINDOW = metadata["acute_window"]
CHRONIC_WINDOW = metadata["chronic_window"]
HRV_BASELINE_WINDOW = metadata["hrv_baseline_window"]
LOW_READINESS_STD_THRESHOLD = metadata["low_readiness_std_threshold"]
PRIOR_VAR = metadata["prior_var"]
N_POST_SAMPLES = metadata["n_post_samples"]
FEATURE_COLS = metadata["feature_cols"]

np.random.seed(metadata["seed"])

# ============================================================
# 2. FEATURE ENGINEERING + LABELING
# ============================================================

def add_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["acute_load"] = out["load"].rolling(ACUTE_WINDOW, min_periods=ACUTE_WINDOW).sum()
    out["chronic_load"] = out["load"].rolling(CHRONIC_WINDOW, min_periods=CHRONIC_WINDOW).sum()
    out["load_ratio"] = out["acute_load"] / (out["chronic_load"] + 1e-8)

    out["hrv_baseline_mean"] = (
        out["hrv"].shift(1)
        .rolling(HRV_BASELINE_WINDOW, min_periods=HRV_BASELINE_WINDOW)
        .mean()
    )
    out["hrv_baseline_std"] = (
        out["hrv"].shift(1)
        .rolling(HRV_BASELINE_WINDOW, min_periods=HRV_BASELINE_WINDOW)
        .std()
    )

    out["overnight_hrv_baseline_mean"] = (
        out["overnight_hrv"].shift(1)
        .rolling(HRV_BASELINE_WINDOW, min_periods=HRV_BASELINE_WINDOW)
        .mean()
    )
    out["overnight_hrv_baseline_std"] = (
        out["overnight_hrv"].shift(1)
        .rolling(HRV_BASELINE_WINDOW, min_periods=HRV_BASELINE_WINDOW)
        .std()
    )

    out["rhr_baseline_mean"] = (
        out["resting_hr"].shift(1)
        .rolling(CHRONIC_WINDOW, min_periods=CHRONIC_WINDOW)
        .mean()
    )

    out["rhr_dev"] = out["resting_hr"] - out["rhr_baseline_mean"]
    out["hrv_dev"] = out["hrv"] - out["hrv_baseline_mean"]
    out["overnight_hrv_dev"] = out["overnight_hrv"] - out["overnight_hrv_baseline_mean"]
    out["overnight_hrv_z"] = (
        out["overnight_hrv"] - out["overnight_hrv_baseline_mean"]
    ) / (out["overnight_hrv_baseline_std"] + 1e-8)

    out["low_readiness_today"] = (
        out["hrv"] < (
            out["hrv_baseline_mean"]
            - LOW_READINESS_STD_THRESHOLD * out["hrv_baseline_std"]
        )
    ).astype(float)

    out["target_next_day"] = out["low_readiness_today"].shift(-1)

    return out

# ============================================================
# 3. BAYESIAN PROBIT MODEL
# ============================================================

def build_model_matrix(df: pd.DataFrame, feature_cols: list[str]):
    model_df = df.dropna(subset=feature_cols + ["target_next_day"]).copy()

    X_raw = model_df[feature_cols].to_numpy(dtype=float)
    y = model_df["target_next_day"].to_numpy(dtype=float)

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0

    X_scaled = (X_raw - X_mean) / X_std
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    return model_df, X, y, X_mean, X_std

def neg_log_posterior(beta: np.ndarray, X: np.ndarray, y: np.ndarray, prior_var: float) -> float:
    eta = X @ beta
    p = norm.cdf(eta)

    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)

    log_like = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    log_prior = -0.5 * np.sum(beta ** 2) / prior_var

    return -(log_like + log_prior)

def numerical_hessian(fun, x0: np.ndarray, eps: float = 1e-4, *args) -> np.ndarray:
    n = len(x0)
    H = np.zeros((n, n))
    f0 = fun(x0, *args)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        f_ip = fun(x0 + ei, *args)
        f_im = fun(x0 - ei, *args)
        H[i, i] = (f_ip - 2 * f0 + f_im) / (eps ** 2)

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            f_pp = fun(x0 + ei + ej, *args)
            f_pm = fun(x0 + ei - ej, *args)
            f_mp = fun(x0 - ei + ej, *args)
            f_mm = fun(x0 - ei - ej, *args)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
            H[i, j] = H_ij
            H[j, i] = H_ij

    return H

def fit_bayesian_probit(X: np.ndarray, y: np.ndarray, prior_var: float = PRIOR_VAR):
    beta0 = np.zeros(X.shape[1])

    result = minimize(
        neg_log_posterior,
        beta0,
        args=(X, y, prior_var),
        method="BFGS",
    )

    if not result.success:
        print("WARNING: Optimization did not fully converge.")
        print("Message:", result.message)

    beta_map = result.x

    H = numerical_hessian(neg_log_posterior, beta_map, 1e-4, X, y, prior_var)
    H += 1e-6 * np.eye(H.shape[0])

    cov_post = np.linalg.inv(H)
    return beta_map, cov_post, result

def posterior_predictive_probability(X: np.ndarray, beta_samples: np.ndarray):
    probs = norm.cdf(X @ beta_samples.T)
    return probs.mean(axis=1), np.quantile(probs, [0.025, 0.975], axis=1)

# ============================================================
# 4. HISTORICAL MODELING
# ============================================================

history_df = add_features_and_labels(history_df)

model_df, X, y, X_mean, X_std = build_model_matrix(history_df, FEATURE_COLS)
beta_map, cov_post, fit_result = fit_bayesian_probit(X, y, PRIOR_VAR)

beta_samples = np.random.multivariate_normal(beta_map, cov_post, size=N_POST_SAMPLES)

pred_mean, pred_ci = posterior_predictive_probability(X, beta_samples)
model_df["pred_prob_next_day"] = pred_mean
model_df["pred_prob_lo"] = pred_ci[0]
model_df["pred_prob_hi"] = pred_ci[1]
model_df["pred_class_05"] = (model_df["pred_prob_next_day"] >= 0.5).astype(int)

feature_names_all = ["intercept"] + FEATURE_COLS
beta_var = np.diag(cov_post)
beta_sd = np.sqrt(beta_var)
beta_lo = beta_map - 1.96 * beta_sd
beta_hi = beta_map + 1.96 * beta_sd

coef_summary = pd.DataFrame({
    "parameter": feature_names_all,
    "posterior_mode_MAP": beta_map,
    "posterior_variance": beta_var,
    "posterior_sd": beta_sd,
    "approx_95pct_CI_low": beta_lo,
    "approx_95pct_CI_high": beta_hi,
})

accuracy = np.mean(model_df["pred_class_05"] == model_df["target_next_day"])
brier = np.mean((model_df["pred_prob_next_day"] - model_df["target_next_day"]) ** 2)

importance_df = pd.DataFrame({
    "parameter": feature_names_all[1:],
    "abs_posterior_mode": np.abs(beta_map[1:]),
    "posterior_sd": beta_sd[1:],
    "posterior_variance": beta_var[1:],
}).sort_values("abs_posterior_mode", ascending=False)

print("\n=================== HISTORICAL DATA SUMMARY ===================")
print(history_df.head(10))
print("\nPhase counts:")
print(history_df["phase"].value_counts())

usable_low_readiness_rate = history_df["low_readiness_today"].dropna().mean()
print(f"\nSame-day low-readiness rate: {usable_low_readiness_rate:.3f}")

print("\n=================== BAYESIAN PROBIT RESULTS ===================")
print(coef_summary.round(4).to_string(index=False))

print("\n=================== FIT QUALITY ===================")
print(f"Usable rows for fitting: {len(model_df)}")
print(f"Observed next-day low-readiness rate: {model_df['target_next_day'].mean():.3f}")
print(f"Classification accuracy at 0.5 threshold: {accuracy:.3f}")
print(f"Brier score: {brier:.3f}")

print("\n=================== VARIABLE IMPORTANCE (STANDARDIZED) ===================")
print(importance_df.round(4).to_string(index=False))

# ============================================================
# 5. FUTURE MODELING
# ============================================================

combined_raw_df = pd.concat([
    history_df[[
        "day", "phase", "race_day", "injury_flag", "miles",
        "load", "sleep_hours", "resting_hr", "hrv", "overnight_hrv"
    ]],
    future_raw_df
], ignore_index=True)

combined_df = add_features_and_labels(combined_raw_df)

max_history_day = history_df["day"].max()
future_df = combined_df[combined_df["day"] > max_history_day].copy()
future_pred_df = future_df.dropna(subset=FEATURE_COLS).copy()

X_future_raw = future_pred_df[FEATURE_COLS].to_numpy(dtype=float)
X_future_scaled = (X_future_raw - X_mean) / X_std
X_future = np.column_stack([np.ones(len(X_future_scaled)), X_future_scaled])

future_pred_mean, future_pred_ci = posterior_predictive_probability(X_future, beta_samples)
future_pred_df["pred_prob_next_day_low_readiness"] = future_pred_mean
future_pred_df["pred_prob_lo"] = future_pred_ci[0]
future_pred_df["pred_prob_hi"] = future_pred_ci[1]
future_pred_df["pred_class_05"] = (
    future_pred_df["pred_prob_next_day_low_readiness"] >= 0.5
).astype(int)

future_accuracy = np.mean(
    future_pred_df["pred_class_05"] == future_pred_df["target_next_day"]
)
future_brier = np.mean(
    (future_pred_df["pred_prob_next_day_low_readiness"] - future_pred_df["target_next_day"]) ** 2
)

print("\n=================== FUTURE EVALUATION ===================")
print(f"Future observed next-day low-readiness rate: {future_pred_df['target_next_day'].mean():.3f}")
print(f"Future accuracy at 0.5 threshold: {future_accuracy:.3f}")
print(f"Future Brier score: {future_brier:.3f}")

print("\n=================== FUTURE PREDICTIONS ===================")
print(
    future_pred_df[[
        "day", "phase", "miles", "load", "sleep_hours",
        "resting_hr", "hrv", "overnight_hrv",
        "target_next_day",
        "pred_prob_next_day_low_readiness",
        "pred_prob_lo", "pred_prob_hi"
    ]].round(3).to_string(index=False)
)

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

history_df.to_csv("garmin_history_with_features.csv", index=False)
model_df.to_csv("historical_modeling_results.csv", index=False)
future_pred_df.to_csv("future_prediction_results.csv", index=False)
coef_summary.to_csv("bayesian_probit_parameter_summary.csv", index=False)
importance_df.to_csv("bayesian_variable_importance.csv", index=False)

print("\nSaved files:")
print("- garmin_history_with_features.csv")
print("- historical_modeling_results.csv")
print("- future_prediction_results.csv")
print("- bayesian_probit_parameter_summary.csv")
print("- bayesian_variable_importance.csv")

# ============================================================
# 7. PLOTS
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(history_df["day"], history_df["miles"], label="Miles")
plt.scatter(
    history_df.loc[history_df["race_day"] == 1, "day"],
    history_df.loc[history_df["race_day"] == 1, "miles"],
    label="Race day",
    s=70,
)
plt.title("Running Season Data")
plt.xlabel("Day")
plt.ylabel("Miles")
plt.legend()
plt.tight_layout()
plt.savefig("Running_Season_Data.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(history_df["day"], history_df["hrv"], label="HRV")
plt.plot(history_df["day"], history_df["overnight_hrv"], label="Overnight HRV")
plt.plot(history_df["day"], history_df["resting_hr"], label="Resting HR")
plt.title("Recovery Metric")
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("Recovery_Metric.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(
    model_df["day"],
    model_df["pred_prob_next_day"],
    label="Predicted probability",
    linewidth=2,
)
plt.fill_between(
    model_df["day"],
    model_df["pred_prob_lo"],
    model_df["pred_prob_hi"],
    alpha=0.2,
    label="95% CI",
)
plt.scatter(
    model_df.loc[model_df["target_next_day"] == 1, "day"],
    model_df.loc[model_df["target_next_day"] == 1, "pred_prob_next_day"],
    color="red",
    label="Observed low readiness",
    s=25,
)
plt.xlabel("Day")
plt.ylabel("Probability")
plt.title("Posterior Predictive Probability Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("Posterior_TimeSeries.png", dpi=300)
plt.show()

def plot_calibration(model_df: pd.DataFrame):
    df = model_df.copy()
    bins = np.linspace(0, 1, 8)
    df["prob_bin"] = pd.cut(df["pred_prob_next_day"], bins, include_lowest=True)

    calib = df.groupby("prob_bin", observed=False).agg(
        mean_pred=("pred_prob_next_day", "mean"),
        observed=("target_next_day", "mean"),
        count=("target_next_day", "count"),
    ).dropna()

    plt.figure(figsize=(6, 6))
    plt.plot(calib["mean_pred"], calib["observed"], marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Calibration.png", dpi=300)
    plt.show()

plot_calibration(model_df)

plt.figure(figsize=(8, 4))
plt.hist(
    model_df.loc[model_df["target_next_day"] == 0, "pred_prob_next_day"],
    bins=25,
    alpha=0.6,
    label="Normal days",
)
plt.hist(
    model_df.loc[model_df["target_next_day"] == 1, "pred_prob_next_day"],
    bins=25,
    alpha=0.6,
    label="Low readiness days",
)
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Probability Distribution by Class")
plt.legend()
plt.tight_layout()
plt.savefig("Prob_Distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(
    future_pred_df["day"],
    future_pred_df["pred_prob_next_day_low_readiness"],
    marker="o",
    label="Predicted probability",
)
plt.fill_between(
    future_pred_df["day"],
    future_pred_df["pred_prob_lo"],
    future_pred_df["pred_prob_hi"],
    alpha=0.2,
    label="95% CI",
)
plt.scatter(
    future_pred_df.loc[future_pred_df["target_next_day"] == 1, "day"],
    future_pred_df.loc[future_pred_df["target_next_day"] == 1, "pred_prob_next_day_low_readiness"],
    color="red",
    s=45,
    label="Actual low-readiness event",
)
plt.title("Future Predictions vs Actual Low-Readiness Events")
plt.xlabel("Day")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("Future_Predictions.png", dpi=300)
plt.show()

# ============================================================
# 8. MARGINAL POSTERIOR DISTRIBUTIONS
# ============================================================

x_grid_std = 4.0

for i, param_name in enumerate(feature_names_all):
    mu_i = beta_map[i]
    sd_i = np.sqrt(cov_post[i, i])

    x_grid = np.linspace(mu_i - x_grid_std * sd_i, mu_i + x_grid_std * sd_i, 400)
    pdf_vals = norm.pdf(x_grid, loc=mu_i, scale=sd_i)

    plt.figure(figsize=(7, 4))
    plt.plot(x_grid, pdf_vals)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(f"Marginal Posterior: {param_name}")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"posterior_{param_name}.png", dpi=300)
    plt.show()