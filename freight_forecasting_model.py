import os, warnings, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import joblib

# CONFIG
BASE_DIR   = r"c:/Users/hp/OneDrive/Desktop/Datasets 06"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ["HSI", "SI", "PI", "CI"]
TARGET_NAMES = {
    "HSI": "Handysize Index",
    "SI":  "Supramax Index",
    "PI":  "Panamax Index",
    "CI":  "Capesize Index",
}
FORECAST_HORIZON = 90

print("=" * 70)
print("  FREIGHT FORECASTING MODEL - TRAINING PIPELINE")
print("=" * 70)

# ========== 1. LOAD DATASETS ==========
print("\n[1/6] Loading datasets...")

# A) Baltic Dry Index (core target)
bdi_path = os.path.join(BASE_DIR, "misc_data", "edited BDI data1.xls")
bdi = pd.read_excel(bdi_path)
bdi["Date"] = pd.to_datetime(bdi["Date"], format="mixed")
bdi = bdi.sort_values("Date").reset_index(drop=True)
bdi = bdi.dropna(subset=["Date"])
bdi = bdi.set_index("Date").asfreq("D").ffill().bfill().reset_index()
bdi.rename(columns={"index": "Date"}, inplace=True)
print(f"   BDI data       : {bdi.shape[0]:,} rows  |  {bdi['Date'].min().date()} to {bdi['Date'].max().date()}")

# B) Commodity Prices
comm_path = os.path.join(BASE_DIR, "commodity_prices_supply_chain.csv")
comm = pd.read_csv(comm_path)
comm["date"] = pd.to_datetime(comm["date"])
comm_pivot = comm.pivot_table(index="date", columns="commodity", values="price", aggfunc="mean")
comm_pivot = comm_pivot.asfreq("D").ffill().bfill()
comm_pivot.columns = ["comm_" + c.replace(" ", "_") for c in comm_pivot.columns]
print(f"   Commodity data  : {comm_pivot.shape[0]:,} rows x {comm_pivot.shape[1]} commodities")

# C) Ship Specifications
ships_path = os.path.join(BASE_DIR, "Cleaned_ships_data.csv")
ships = pd.read_csv(ships_path)
def classify_vessel(dwt):
    if dwt < 40000:  return "Handysize"
    elif dwt < 60000: return "Supramax"
    elif dwt < 100000: return "Panamax"
    else: return "Capesize"
ships["vessel_class"] = ships["dwt"].apply(classify_vessel)
ship_stats = ships.groupby("vessel_class").agg(
    avg_dwt=("dwt", "mean"), avg_gt=("gt", "mean"),
    avg_length=("length", "mean"), avg_width=("width", "mean"),
    count=("dwt", "count")
).reset_index()
print(f"   Ship specs      : {ships.shape[0]} vessels in {ships['vessel_class'].nunique()} classes")
print(ship_stats.to_string(index=False))

# D) Visakhapatnam Port Traffic
vizag_path = os.path.join(BASE_DIR, "TableNo11_TRAFFIC_HANDLED_AT_VISHAKHAPATNAM_PORT.csv")
vizag = pd.read_csv(vizag_path)
vizag["start_year"] = vizag["Year/Ports"].str.extract(r"(\d{4})").astype(float)
vizag = vizag.dropna(subset=["start_year"])
vizag["start_year"] = vizag["start_year"].astype(int)
for col in ["Total Traffic", "Overseas Total", "Coastal Total"]:
    vizag[col] = pd.to_numeric(vizag[col], errors="coerce")
print(f"   Vizag traffic   : {vizag.shape[0]} years")

# E) Seaborne Trade
sbt_path = os.path.join(BASE_DIR, "US_SeaborneTrade_large", "US.SeaborneTrade_20260920_141012.csv")
sbt = pd.read_csv(sbt_path)
sbt_pivot = sbt.pivot_table(index="Year", columns="CargoType_Label",
    values="Metric_tons_in_thousands_Value", aggfunc="sum")
sbt_pivot.columns = ["sbt_" + c.replace(" ", "_").replace("/", "_") for c in sbt_pivot.columns]
print(f"   Seaborne trade  : {sbt_pivot.shape[0]} years x {sbt_pivot.shape[1]} cargo types")

# F) Port Calls
pc_path = os.path.join(BASE_DIR, "US_PortCalls", "US.PortCalls_20260920_140718.csv")
port_calls = pd.read_csv(pc_path)
print(f"   Port calls      : {port_calls.shape[0]} rows x {port_calls.shape[1]} cols")

# G) Transport Costs
tc_path = os.path.join(BASE_DIR, "US_TransportCosts", "US.TransportCosts_20260920_141624.csv")
transport_costs = pd.read_csv(tc_path)
tc_years = [c for c in transport_costs.columns if "Perunit_freight_rate" in c and "Value" in c]
tc_world = transport_costs[transport_costs["Destination_Label"].str.contains("World", na=False)]
if not tc_world.empty:
    tc_series = tc_world[tc_years].iloc[0].values.astype(float)
    tc_year_labels = [int(c.split("_")[0]) for c in tc_years]
    tc_df = pd.DataFrame({"year": tc_year_labels, "world_freight_rate_usd_kg": tc_series})
    print(f"   Transport costs : {len(tc_df)} years of world freight-rate data")
else:
    tc_df = pd.DataFrame()
    print("   Transport costs : World row not found, skipping")

# ========== 2. FEATURE ENGINEERING ==========
print("\n[2/6] Engineering features...")
df = bdi.copy()

# A) Lagged features
for t in TARGETS:
    for lag in [1, 3, 7, 14, 30]:
        df[f"{t}_lag{lag}"] = df[t].shift(lag)
    for win in [7, 14, 30]:
        df[f"{t}_roll_mean{win}"]  = df[t].rolling(win).mean()
        df[f"{t}_roll_std{win}"]   = df[t].rolling(win).std()
        df[f"{t}_roll_min{win}"]   = df[t].rolling(win).min()
        df[f"{t}_roll_max{win}"]   = df[t].rolling(win).max()
    df[f"{t}_roc7"]  = df[t].pct_change(7)
    df[f"{t}_roc30"] = df[t].pct_change(30)

# B) Calendar features
df["day_of_week"]  = df["Date"].dt.dayofweek
df["month"]        = df["Date"].dt.month
df["quarter"]      = df["Date"].dt.quarter
df["day_of_year"]  = df["Date"].dt.dayofyear
df["year"]         = df["Date"].dt.year
df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
df["doy_sin"]      = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["doy_cos"]      = np.cos(2 * np.pi * df["day_of_year"] / 365)

# C) Cross-index ratios
df["CI_PI_ratio"]  = df["CI"] / (df["PI"] + 1)
df["PI_SI_ratio"]  = df["PI"] / (df["SI"] + 1)
df["SI_HSI_ratio"] = df["SI"] / (df["HSI"] + 1)

# D) Merge commodity prices
df = df.merge(comm_pivot, left_on="Date", right_index=True, how="left")
comm_cols = [c for c in df.columns if c.startswith("comm_")]
df[comm_cols] = df[comm_cols].ffill().bfill()

key_commodities = ["comm_Thermal_Coal_Newcastle", "comm_Brent_Crude",
                   "comm_WTI_Crude", "comm_Corn", "comm_Wheat", "comm_Copper"]
for kc in key_commodities:
    if kc in df.columns:
        df[f"{kc}_lag7"]  = df[kc].shift(7)
        df[f"{kc}_lag30"] = df[kc].shift(30)
        df[f"{kc}_roc7"]  = df[kc].pct_change(7)

# E) Transport cost
if not tc_df.empty:
    df = df.merge(tc_df, on="year", how="left")
    df["world_freight_rate_usd_kg"] = df["world_freight_rate_usd_kg"].ffill().bfill()

# F) Seaborne trade
sbt_yearly = sbt_pivot.reset_index()
sbt_yearly.rename(columns={"Year": "year"}, inplace=True)
df = df.merge(sbt_yearly, on="year", how="left")
sbt_cols = [c for c in df.columns if c.startswith("sbt_")]
df[sbt_cols] = df[sbt_cols].ffill().bfill()

# G) Vizag traffic
vizag_clean = vizag[["start_year", "Total Traffic"]].dropna()
vizag_clean.rename(columns={"start_year": "year", "Total Traffic": "vizag_traffic"}, inplace=True)
df = df.merge(vizag_clean, on="year", how="left")
df["vizag_traffic"] = df["vizag_traffic"].ffill().bfill()

df = df.dropna()
print(f"   Final dataset   : {df.shape[0]:,} rows x {df.shape[1]} features")

# ========== 3. TRAIN/TEST SPLIT ==========
print("\n[3/6] Preparing train/test splits...")
exclude_cols = ["Date"] + TARGETS + ["DTI", "CTI"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()
print(f"   Train: {train_df.shape[0]:,} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
print(f"   Test:  {test_df.shape[0]:,} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

# ========== 4. TRAIN MODELS ==========
print("\n[4/6] Training models for each freight sub-index...")
results = {}
best_models = {}

for target in TARGETS:
    print(f"\n  -- {TARGET_NAMES[target]} ({target}) --")
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Ridge":            Ridge(alpha=10),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=15,
                                                   min_samples_leaf=5, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=6,
                                                       learning_rate=0.05, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, random_state=42, verbosity=0)

    best_score = float("inf")
    best_name  = None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        mape  = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100
        print(f"     {name:20s}  MAE={mae:8.1f}  RMSE={rmse:8.1f}  R2={r2:.4f}  MAPE={mape:.2f}%")
        if mae < best_score:
            best_score = mae
            best_name  = name
            best_model = model
            best_preds = preds

    print(f"     * Best model: {best_name} (MAE={best_score:.1f})")
    results[target] = {
        "best_model": best_name,
        "MAE":  round(best_score, 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, best_preds)), 2),
        "R2":   round(r2_score(y_test, best_preds), 4),
        "MAPE": round(np.mean(np.abs((y_test - best_preds) / (y_test + 1e-8))) * 100, 2),
    }
    best_models[target] = {"model": best_model, "scaler": scaler, "name": best_name}

    model_file = os.path.join(OUTPUT_DIR, f"{target}_best_model.joblib")
    joblib.dump({"model": best_model, "scaler": scaler, "features": feature_cols, "name": best_name}, model_file)

    pred_df = test_df[["Date", target]].copy()
    pred_df["predicted"] = best_preds
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"{target}_test_predictions.csv"), index=False)

# ========== 5. 90-DAY FORECAST ==========
print("\n[5/6] Generating 90-day future forecasts...")
last_date = df["Date"].iloc[-1]
forecast_rows = []
current_data = df.tail(60).copy()

for day in range(1, FORECAST_HORIZON + 1):
    new_date = last_date + timedelta(days=day)
    new_row = current_data.iloc[-1:].copy()
    new_row["Date"] = new_date
    new_row["day_of_week"] = new_date.dayofweek
    new_row["month"]       = new_date.month
    new_row["quarter"]     = new_date.quarter
    new_row["day_of_year"] = new_date.timetuple().tm_yday
    new_row["year"]        = new_date.year
    new_row["month_sin"]   = np.sin(2 * np.pi * new_date.month / 12)
    new_row["month_cos"]   = np.cos(2 * np.pi * new_date.month / 12)
    new_row["doy_sin"]     = np.sin(2 * np.pi * new_date.timetuple().tm_yday / 365)
    new_row["doy_cos"]     = np.cos(2 * np.pi * new_date.timetuple().tm_yday / 365)

    day_forecast = {"Date": new_date}
    for target in TARGETS:
        bm = best_models[target]
        X_new = new_row[feature_cols].values
        X_new_s = bm["scaler"].transform(X_new)
        pred = max(bm["model"].predict(X_new_s)[0], 0)
        day_forecast[target] = round(pred, 1)
        new_row[target] = pred
        for lag in [1, 3, 7, 14, 30]:
            col = f"{target}_lag{lag}"
            if col in new_row.columns and lag <= len(current_data):
                new_row[col] = current_data[target].iloc[-lag]

    forecast_rows.append(day_forecast)
    current_data = pd.concat([current_data, new_row], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(os.path.join(OUTPUT_DIR, "freight_90day_forecast.csv"), index=False)
print(f"   Saved 90-day forecast to {OUTPUT_DIR}")

# ========== 6. VESSEL RECOMMENDATION ENGINE ==========
print("\n[6/6] Building Vessel Recommendation Engine...")

port_constraints = {
    "Paradip":      {"max_draft_m": 14.5, "max_loa_m": 300, "max_beam_m": 50, "handling_rate_tpd": 35000},
    "Vizag":        {"max_draft_m": 18.1, "max_loa_m": 350, "max_beam_m": 55, "handling_rate_tpd": 45000},
    "Gangavaram":   {"max_draft_m": 21.0, "max_loa_m": 330, "max_beam_m": 57, "handling_rate_tpd": 50000},
    "Gopalpur":     {"max_draft_m": 14.0, "max_loa_m": 230, "max_beam_m": 38, "handling_rate_tpd": 15000},
    "Dhamra":       {"max_draft_m": 18.0, "max_loa_m": 330, "max_beam_m": 55, "handling_rate_tpd": 40000},
    "Sagar_Sandheads": {"max_draft_m": 9.0,  "max_loa_m": 200, "max_beam_m": 30, "handling_rate_tpd": 10000},
    "Haldia":       {"max_draft_m": 9.5,  "max_loa_m": 200, "max_beam_m": 32, "handling_rate_tpd": 12000},
}

vessel_classes = {
    "Handysize":  {"dwt_range": (15000, 40000),  "typ_draft_m": 10, "typ_loa_m": 180, "typ_beam_m": 30},
    "Supramax":   {"dwt_range": (40000, 60000),  "typ_draft_m": 12, "typ_loa_m": 200, "typ_beam_m": 32},
    "Panamax":    {"dwt_range": (60000, 100000), "typ_draft_m": 14, "typ_loa_m": 230, "typ_beam_m": 32},
    "Capesize":   {"dwt_range": (100000, 400000),"typ_draft_m": 18, "typ_loa_m": 300, "typ_beam_m": 50},
}

index_to_class = {"HSI": "Handysize", "SI": "Supramax", "PI": "Panamax", "CI": "Capesize"}

def recommend_vessel(cargo_mt, dest_port):
    port = port_constraints.get(dest_port)
    if not port:
        return f"Port '{dest_port}' not found"
    recs = []
    for vc_name, vc_spec in vessel_classes.items():
        fits = (vc_spec["typ_draft_m"] <= port["max_draft_m"] and
                vc_spec["typ_loa_m"]   <= port["max_loa_m"] and
                vc_spec["typ_beam_m"]  <= port["max_beam_m"])
        if not fits:
            continue
        min_dwt, max_dwt = vc_spec["dwt_range"]
        voyages = max(1, int(np.ceil(cargo_mt / max_dwt)))
        turnaround = cargo_mt / voyages / port["handling_rate_tpd"]
        idx = [k for k, v in index_to_class.items() if v == vc_name][0]
        avg_fc = forecast_df[idx].mean()
        recs.append({
            "vessel_class": vc_name, "voyages_needed": voyages,
            "cargo_per_voyage_mt": round(cargo_mt / voyages),
            "est_turnaround_days": round(turnaround, 1),
            "avg_forecast_index": round(avg_fc, 1), "port_fit": True,
        })
    recs.sort(key=lambda x: (x["voyages_needed"], x["avg_forecast_index"]))
    return recs

print("\n  Example: 150,000 MT coal to Vizag:")
for r in recommend_vessel(150000, "Vizag"):
    print(f"    {r['vessel_class']:12s} | {r['voyages_needed']} voyage(s) | ~{r['cargo_per_voyage_mt']:,} MT/voy | ~{r['est_turnaround_days']}d turnaround | Forecast idx: {r['avg_forecast_index']}")

print("\n  Example: 50,000 MT coal to Haldia:")
for r in recommend_vessel(50000, "Haldia"):
    print(f"    {r['vessel_class']:12s} | {r['voyages_needed']} voyage(s) | ~{r['cargo_per_voyage_mt']:,} MT/voy | ~{r['est_turnaround_days']}d turnaround | Forecast idx: {r['avg_forecast_index']}")

# ========== 7. SAVE REPORT ==========
report = {
    "model_performance": results,
    "datasets_used": {
        "BDI_freight_indices": f"{bdi.shape[0]} daily records",
        "commodity_prices": f"{comm_pivot.shape[0]} daily records, {comm_pivot.shape[1]} commodities",
        "ship_specs": f"{ships.shape[0]} vessels",
        "vizag_traffic": f"{vizag.shape[0]} yearly records",
        "seaborne_trade": f"{sbt_pivot.shape[0]} yearly records",
        "transport_costs": f"{len(tc_df)} yearly records",
        "port_calls": f"{port_calls.shape[0]} records",
    },
    "feature_count": len(feature_cols),
    "train_size": train_df.shape[0],
    "test_size": test_df.shape[0],
    "forecast_horizon_days": FORECAST_HORIZON,
    "port_constraints": port_constraints,
}
with open(os.path.join(OUTPUT_DIR, "model_report.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)

pd.DataFrame(port_constraints).T.to_csv(os.path.join(OUTPUT_DIR, "port_constraints.csv"))
ship_stats.to_csv(os.path.join(OUTPUT_DIR, "vessel_class_stats.csv"), index=False)

print("\n" + "=" * 70)
print("  TRAINING COMPLETE - RESULTS SUMMARY")
print("=" * 70)
print(f"\n  {'Index':<25s} {'Best Model':<22s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'MAPE':>8s}")
print("  " + "-" * 73)
for t in TARGETS:
    r = results[t]
    print(f"  {TARGET_NAMES[t]:<25s} {r['best_model']:<22s} {r['MAE']:>8.1f} {r['RMSE']:>8.1f} {r['R2']:>8.4f} {r['MAPE']:>7.2f}%")

print(f"\n  Forecast preview (next 7 days from {last_date.date()}):")
print(f"  {'Date':<12s}  {'HSI':>8s}  {'SI':>8s}  {'PI':>8s}  {'CI':>8s}")
print("  " + "-" * 44)
for _, row in forecast_df.head(7).iterrows():
    print(f"  {str(row['Date'].date()):<12s}  {row['HSI']:>8.1f}  {row['SI']:>8.1f}  {row['PI']:>8.1f}  {row['CI']:>8.1f}")

print(f"\n  All outputs saved to: {OUTPUT_DIR}")
print("  Files: HSI/SI/PI/CI_best_model.joblib, *_test_predictions.csv,")
print("         freight_90day_forecast.csv, model_report.json,")
print("         port_constraints.csv, vessel_class_stats.csv")
print("=" * 70)
