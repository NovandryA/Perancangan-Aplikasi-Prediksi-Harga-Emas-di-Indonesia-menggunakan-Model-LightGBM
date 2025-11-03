# train_single_shot.py
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

# Konfigurasi
DATASET_PATH = "Dataset_HargaEmas.xlsx"   
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

USE_LOG_FOR = [7]    

# Load & Clean data
df = pd.read_excel(DATASET_PATH)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.sort_values(date_col).ffill().bfill()
df = df[df[date_col].notna()].reset_index(drop=True)

assert "Gold_Price" in df.columns, "Kolom Gold_Price tidak ditemukan!"
macro_cols = [c for c in ["USD_Buy_Rate","USD_Sell_Rate","BI_Rate"] if c in df.columns]

df["Month"]     = df[date_col].dt.month
df["DayOfWeek"] = df[date_col].dt.dayofweek
for col in ["Gold_Price"] + macro_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[["Gold_Price"] + macro_cols] = df[["Gold_Price"] + macro_cols].ffill().bfill()

deploy_features = ["Month","DayOfWeek","Gold_Price"] + macro_cols

def make_target_and_reconstruct(frame: pd.DataFrame, N: int, use_log: bool):
    eps = 1e-9
    if use_log:
        target = np.log(frame["Gold_Price"].shift(-N) + eps) - np.log(frame["Gold_Price"] + eps)
        def reconstruct(pred, P_t):
            P_next = np.exp(pred) * P_t
            return P_next, (P_next - P_t)
    else:
        target = frame["Gold_Price"].shift(-N) - frame["Gold_Price"]
        def reconstruct(pred, P_t):
            P_next = P_t + pred
            return P_next, pred
    return target, reconstruct

def train_deltaN(N: int):
    use_log = (N in USE_LOG_FOR)

    dataN = df.copy()
    y_series, reconstruct = make_target_and_reconstruct(dataN, N, use_log)
    dataN["Target"] = pd.to_numeric(y_series, errors="coerce")

    
    features = [c for c in deploy_features if c in dataN.columns]
    for c in features + ["Target","Gold_Price"]:
        dataN[c] = pd.to_numeric(dataN[c], errors="coerce")

    
    dataN = dataN.dropna(subset=features + ["Target","Gold_Price"]).reset_index(drop=True)

    X_full = dataN[features]
    y_full = dataN["Target"]
    split_idx = int(len(dataN)*0.8)
    X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
    P_t_test = dataN["Gold_Price"].iloc[split_idx:].to_numpy()

    # --- Bayesian Optimization untuk hyperparameter tuning ---
    def cv_rmse(num_leaves,max_depth,feature_fraction,min_child_samples,
                learning_rate,n_estimators,top_rate,other_rate,max_bin,min_data_in_bin):
        params = dict(
            objective="regression", boosting_type="goss", metric="rmse",
            num_leaves=int(num_leaves), max_depth=int(max_depth),
            feature_fraction=float(feature_fraction),
            min_child_samples=int(min_child_samples),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            top_rate=float(top_rate), other_rate=float(other_rate),
            max_bin=int(max_bin), min_data_in_bin=int(min_data_in_bin),
            reg_alpha=0.0, reg_lambda=0.0, verbose=-1
        )
        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        for tr, va in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            m = LGBMRegressor(**params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)]
            )
            preds = m.predict(X_va)
            rmse = np.sqrt(((y_va - preds)**2).mean())
            rmses.append(rmse)
        return -float(np.mean(rmses))

    from bayes_opt import BayesianOptimization
    lr_low  = 0.01 if not use_log else 0.005
    lr_high = 0.15 if not use_log else 0.10
    pbounds = {
        "num_leaves": (63,255), "max_depth": (4,16),
        "feature_fraction": (0.6,1.0), "min_child_samples": (10,180),
        "learning_rate": (lr_low, lr_high), "n_estimators": (800,3000),
        "top_rate": (0.3,0.6), "other_rate": (0.1,0.3),
        "max_bin": (255,1023), "min_data_in_bin": (1,20)
    }
    n_iter = 30 if not use_log and N < 6 else 50
    opt = BayesianOptimization(f=cv_rmse, pbounds=pbounds, random_state=42, verbose=0)
    opt.maximize(init_points=10, n_iter=n_iter)
    best = opt.max["params"]
    for k in ["num_leaves","max_depth","min_child_samples","n_estimators","max_bin","min_data_in_bin"]:
        best[k] = int(best[k])

    model = LGBMRegressor(objective="regression", boosting_type="goss", metric="rmse", **best)
    val_cut = int(len(X_train)*0.85)
    model.fit(
        X_train.iloc[:val_cut], y_train.iloc[:val_cut],
        eval_set=[(X_train.iloc[val_cut:], y_train.iloc[val_cut:])],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=300), log_evaluation(period=0)]
    )

    # evaluasi delta/log
    y_pred = model.predict(X_test)
    mae_delta  = mean_absolute_error(y_test, y_pred)
    rmse_delta = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_delta   = r2_score(y_test, y_pred)

    # evaluasi level (rekonstruksi)
    P_pred_next, _ = reconstruct(y_pred, P_t_test)
    P_true_next, _ = reconstruct(y_test, P_t_test)
    mae_level  = mean_absolute_error(P_true_next, P_pred_next)
    rmse_level = np.sqrt(mean_squared_error(P_true_next, P_pred_next))
    r2_level   = r2_score(P_true_next, P_pred_next)

    # simpan model
    target_kind = "log" if use_log else "delta"
    out_path = MODELS_DIR / f"Model_LightGBM_GOSS_Delta{N}_{target_kind}.pkl"
    import joblib; joblib.dump(model, out_path)

    # return ringkasan
    return {
        "N": N, "target": target_kind, "features": features,
        "MAE_delta": mae_delta, "RMSE_delta": rmse_delta, "R2_delta": r2_delta,
        "MAE_level": mae_level, "RMSE_level": rmse_level, "R2_level": r2_level
    }

# Train semua horizon
all_rows = []
for N in range(1, 8):
    print(f"Training Δ{N} (single-shot)...")
    res = train_deltaN(N)
    all_rows.append(res)

# model_meta.json
meta = { f"Delta{r['N']}": {"features": r["features"], "target": r["target"]} for r in all_rows }
with open(MODELS_DIR / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# metrics_all_horizons.csv
metrics_df = pd.DataFrame([{
    "Horizon": r["N"], "MAE_level": r["MAE_level"], "RMSE_level": r["RMSE_level"], "R2_level": r["R2_level"],
    "MAE_delta": r["MAE_delta"], "RMSE_delta": r["RMSE_delta"], "R2_delta": r["R2_delta"],
    "Target": r["target"]
} for r in all_rows]).sort_values("Horizon")
metrics_df.to_csv(MODELS_DIR / "metrics_all_horizons.csv", index=False)

print("\nSelesai. Model & meta disimpan di folder 'models/'.")
print(metrics_df[["Horizon","Target","MAE_level","RMSE_level","R2_level"]])