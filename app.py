import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quieter TF logs
# If you want to disable oneDNN message:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify, send_from_directory

import numpy as np
import datetime as dt
import joblib

# Keras/TensorFlow (compile=False to avoid KerasSaveable errors)
try:
    from keras.models import load_model  # type: ignore[import-untyped]
except Exception:
    load_model = None

app = Flask(__name__)

# -----------------------------
# Optional model loading
# -----------------------------
def load_iso_forest():
    for p in ["isolation_forest_model.pkl", os.path.join("models", "isolation_forest_model.pkl")]:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                print(f"✅ IsolationForest loaded: {p}")
                return m
            except Exception as e:
                print(f"⚠️ Failed to load IsolationForest from {p}: {e}")
    print("ℹ️ IsolationForest not found; anomaly endpoint will be disabled.")
    return None

def load_lstm():
    if load_model is None:
        print("ℹ️ Keras not available; using fallback predictor.")
        return None
    for p in ["lstm_pollution_model.h5", os.path.join("models", "lstm_pollution_model.h5")]:
        if os.path.exists(p):
            try:
                m = load_model(p, compile=False)
                # compile not required for inference, but harmless if later evaluated
                try:
                    m.compile(optimizer="adam", loss="mse")
                except Exception:
                    pass
                print(f"✅ LSTM loaded: {p}")
                return m
            except Exception as e:
                print(f"⚠️ Failed to load LSTM from {p}: {e}")
    print("ℹ️ LSTM not found; using fallback predictor.")
    return None

iso_forest = load_iso_forest()
lstm_model = load_lstm()

# -----------------------------
# Utilities & domain logic
# -----------------------------
def classify_pollutants(pm25, pm10, co2):
    # PM2.5
    if pm25 <= 30: pm25_status = "Good"
    elif pm25 <= 60: pm25_status = "Satisfactory"
    elif pm25 <= 90: pm25_status = "Moderate"
    elif pm25 <= 120: pm25_status = "Poor"
    elif pm25 <= 250: pm25_status = "Very Poor"
    else: pm25_status = "Severe"
    # PM10
    if pm10 <= 50: pm10_status = "Good"
    elif pm10 <= 100: pm10_status = "Satisfactory"
    elif pm10 <= 250: pm10_status = "Moderate"
    elif pm10 <= 350: pm10_status = "Poor"
    elif pm10 <= 430: pm10_status = "Very Poor"
    else: pm10_status = "Severe"
    # CO2
    if co2 <= 450: co2_status = "Good"
    elif co2 <= 1000: co2_status = "Satisfactory"
    elif co2 <= 2000: co2_status = "Moderate"
    elif co2 <= 5000: co2_status = "Poor"
    else: co2_status = "Severe"

    return {
        "PM2.5": {"value": float(pm25), "status": pm25_status},
        "PM10":  {"value": float(pm10),  "status": pm10_status},
        "CO2":   {"value": float(co2),   "status": co2_status},
    }

def seeded_rng(lat, lon):
    # Convert lat/lon to integers before XOR
    lat_int = int(float(lat) * 1000)
    lon_int = int(float(lon) * 1000)
    seed = lat_int ^ lon_int
    rng = np.random.default_rng(seed)
    return rng


def simulate_base_levels(rng):
    # base typical urban baselines
    base_pm25 = rng.normal(80, 15)     # µg/m³
    base_pm10 = rng.normal(150, 30)    # µg/m³
    base_co2  = rng.normal(900, 120)   # ppm (outdoor ref-ish for demo)
    return max(base_pm25, 5), max(base_pm10, 10), max(base_co2, 350)

def make_72h_history(lat, lon):
    rng = seeded_rng(lat, lon)
    pm25_base, pm10_base, co2_base = simulate_base_levels(rng)

    now = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now - dt.timedelta(hours=h) for h in range(71, -1, -1)]  # oldest→newest

    history = []
    for i, t in enumerate(times):
        # gentle diurnal cycles + noise
        diurnal = np.sin(2*np.pi*(t.hour/24))
        pm25 = pm25_base * (1 + 0.15*diurnal) + rng.normal(0, 8)
        pm10 = pm10_base * (1 + 0.12*diurnal) + rng.normal(0, 15)
        co2  = co2_base  * (1 + 0.05*diurnal) + rng.normal(0, 35)
        history.append({
            "timestamp": t.isoformat(),
            "PM2_5": round(max(pm25, 1), 2),
            "PM10":  round(max(pm10, 1), 2),
            "CO2":   round(max(co2, 300), 2),
        })
    return history

def lstm_predict_next_72(lat, lon, history):
    """
    If LSTM exists, try to use it. Expected input: (1, T, 3) with order [PM2.5, PM10, CO2].
    If absent or fails, produce AR(1)-ish forecast from history tail.
    """
    # Build past array (T, 3) → use last min(72, len(history)) points
    tail = history[-72:]
    base_past = np.array([[h["PM2_5"], h["PM10"], h["CO2"]] for h in tail], dtype=np.float32)  # (72,3)

    # If we have an LSTM, adapt the input to the model's expected shape (timesteps, features).
    if lstm_model is not None:
        try:
            # Inspect expected input shape if available
            try:
                inp_shape = lstm_model.input_shape  # typically (None, T, F) or (1, T, F)
            except Exception:
                inp_shape = None

            past = base_past
            # Determine target timesteps and features
            target_timesteps = None
            target_features = None
            if inp_shape and len(inp_shape) >= 3:
                target_timesteps = inp_shape[1]
                target_features = inp_shape[2]

            # Adjust timesteps: trim or pad (repeat first row) to match expected length
            if target_timesteps is not None:
                cur_t = past.shape[0]
                if cur_t > target_timesteps:
                    past = past[-target_timesteps:]
                elif cur_t < target_timesteps:
                    # pad at start with first row repeated
                    pad = np.tile(past[0:1, :], (target_timesteps - cur_t, 1))
                    past = np.vstack([pad, past])

            # Adjust feature dimension: if model expects more than 3 features, append engineered features
            if target_features is None:
                x = past[np.newaxis, ...]
            else:
                if target_features <= 3:
                    # truncate to requested features if necessary
                    past = past[:, :target_features]
                    x = past[np.newaxis, ...]
                else:
                    # Prepare deterministic extra features per timestamp
                    extras = []
                    for h in (tail if target_timesteps is None or len(tail) == past.shape[0] else history[-past.shape[0]:]):
                        ts = dt.datetime.fromisoformat(h["timestamp"])
                        hour = ts.hour
                        dow = ts.weekday()
                        hour_norm = hour / 23.0
                        sin_h = np.sin(2 * np.pi * (hour / 24.0))
                        cos_h = np.cos(2 * np.pi * (hour / 24.0))
                        dow_norm = dow / 6.0
                        sin_d = np.sin(2 * np.pi * (dow / 7.0))
                        cos_d = np.cos(2 * np.pi * (dow / 7.0))
                        extras.append([hour_norm, sin_h, cos_h, dow_norm, sin_d, cos_d])
                    extras = np.array(extras, dtype=np.float32)

                    # rolling means (3-hr) for PM2.5 and PM10
                    rm = []
                    for i in range(past.shape[0]):
                        start = max(0, i - 2)
                        rm_pm25 = past[start:i+1, 0].mean()
                        rm_pm10 = past[start:i+1, 1].mean()
                        rm.append([rm_pm25, rm_pm10])
                    rm = np.array(rm, dtype=np.float32)

                    # Compose candidate extra columns in a stable order
                    candidate_extras = np.concatenate([extras, rm], axis=1)  # shape (T, 8)

                    need = target_features - 3
                    if need > candidate_extras.shape[1]:
                        # If still short, pad with zeros
                        pad_cols = np.zeros((candidate_extras.shape[0], need - candidate_extras.shape[1]), dtype=np.float32)
                        add = np.concatenate([candidate_extras, pad_cols], axis=1)[:, :need]
                    else:
                        add = candidate_extras[:, :need]

                    past = np.concatenate([past, add], axis=1)
                    x = past[np.newaxis, ...]

            y = lstm_model.predict(x)
            y = np.array(y)
            # Normalize shapes to (72,3)
            if y.ndim == 3 and y.shape[0] == 1:
                pred = y[0]
                # if not 72, resize with repeat (demo handling)
                if pred.shape[0] != 72:
                    # naive repeat/trim to 72 steps
                    rep = int(np.ceil(72 / pred.shape[0]))
                    pred = np.tile(pred, (rep, 1))[:72, :]
            elif y.ndim == 2:
                # (1, T) or (T, 3) or (1, 3)
                if y.shape[0] == 1:  # (1, T) or (1, 3)
                    arr = y[0]
                else:
                    arr = y
                arr = np.array(arr)
                if arr.ndim == 1:  # (T,)
                    arr = arr[:, None]
                if arr.shape[1] < 3:
                    pad = np.full((arr.shape[0], 3 - arr.shape[1]), np.nan, dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=1)
                pred = arr
                # Repeat/trim to 72
                rep = int(np.ceil(72 / pred.shape[0]))
                pred = np.tile(pred, (rep, 1))[:72, :]
            else:
                pred = None
            if pred is not None:
                # replace NaNs with last known values (per column)
                last_vals = past[-1]
                if np.any(np.isnan(pred)):
                    pred = pred.copy()
                    for j in range(pred.shape[1]):
                        col = pred[:, j]
                        col[np.isnan(col)] = last_vals[j]
                return pred
        except Exception as e:
            print("⚠️ LSTM prediction failed; falling back. Reason:", e)

    # Fallback: AR(1)-like on last step + small random walk
    rng = seeded_rng(lat, lon)
    preds = []
    last = past[-1].copy()
    for i in range(72):
        shock = rng.normal([0, 0, 0], [6, 10, 20])
        drift = np.array([-0.1, -0.1, 0.0])  # tiny mean-reversion for PMs
        last = np.maximum(last + drift + shock, [1, 1, 350])
        preds.append(last.copy())
    return np.array(preds, dtype=np.float32)

# Small demo catalog of industries (add your real data here)
INDUSTRY_CATALOG = [
    {"name": "Shiv Shakti Steel", "lat": 25.35, "lon": 82.97, "pollutants": ["PM2.5", "PM10"]},
    {"name": "Varuna Thermal Power", "lat": 25.28, "lon": 83.03, "pollutants": ["PM2.5", "CO2"]},
    {"name": "Ganga Chemicals", "lat": 25.32, "lon": 82.95, "pollutants": ["CO2"]},
    {"name": "Kashi Cement Works", "lat": 25.30, "lon": 82.98, "pollutants": ["PM10"]},
]

def pick_responsible_industry(lat, lon, dominant_pollutant):
    # nearest industry that emits dominant pollutant
    lat, lon = float(lat), float(lon)
    best = None
    best_dist2 = 1e9
    for ind in INDUSTRY_CATALOG:
        if dominant_pollutant not in ind["pollutants"]:
            continue
        d2 = (lat - ind["lat"])**2 + (lon - ind["lon"])**2
        if d2 < best_dist2:
            best_dist2 = d2
            best = ind
    # if none matched, choose nearest overall
    if best is None:
        for ind in INDUSTRY_CATALOG:
            d2 = (lat - ind["lat"])**2 + (lon - ind["lon"])**2
            if d2 < best_dist2:
                best_dist2 = d2
                best = ind
    return best

def summarize_current_status(history_tail):
    """Use the latest observation as current status and classify."""
    latest = history_tail[-1]
    return classify_pollutants(latest["PM2_5"], latest["PM10"], latest["CO2"])

def find_dominant_pollutant(values_dict):
    # values_dict is like {"PM2.5": {...}, "PM10": {...}, "CO2": {...}}
    numeric = {k: v["value"] for k, v in values_dict.items()}
    return max(numeric, key=numeric.get)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("login.html")  # Your login page

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")  # Your dashboard


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "lstm_loaded": lstm_model is not None,
        "iso_forest_loaded": iso_forest is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Request JSON:
    {
      "city": "Varanasi",
      "lat": 25.3176,
      "lon": 82.9739,
      "date": "2025-08-25"  # future date wanted
    }
    Response JSON:
    {
      "city": ...,
      "coords": {"lat":..,"lon":..},
      "history_72h": [{timestamp, PM2_5, PM10, CO2}, ...],
      "current_status": { PM2.5:{value,status}, PM10:{...}, CO2:{...} },
      "future_72h":    [{timestamp, PM2_5, PM10, CO2}, ...],
      "prediction_on_date": {timestamp, PM2_5, PM10, CO2, classified:{...}},
      "industry": {name, lat, lon, pollutants:[...]}
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    city = data.get("city", "Unknown")
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon are required and must be numbers"}), 400
    date_str = data.get("date") or None

    # Build history (72h hourly)
    history = make_72h_history(lat, lon)

    # Current status from the latest history point
    current_status = summarize_current_status(history)

    # Predict next 72h
    next72 = lstm_predict_next_72(lat, lon, history)

    # Compose future timeline
    last_ts = dt.datetime.fromisoformat(history[-1]["timestamp"])
    fut_times = [last_ts + dt.timedelta(hours=i+1) for i in range(72)]
    future = []
    for i, t in enumerate(fut_times):
        row = {
            "timestamp": t.isoformat(),
            "PM2_5": round(float(next72[i, 0]), 2),
            "PM10":  round(float(next72[i, 1]), 2),
            "CO2":   round(float(next72[i, 2]), 2),
        }
        future.append(row)

    # Prediction for requested date (use noon of that day)
    date_pred = None
    if date_str and str(date_str).strip():
        try:
            target = dt.datetime.fromisoformat(str(date_str).strip())
            target_noon = target.replace(hour=12, minute=0, second=0, microsecond=0)
            # If within our future horizon, find nearest future hour; else extrapolate with last step
            if target_noon <= fut_times[-1]:
                idx = int((target_noon - fut_times[0]).total_seconds() // 3600)
                idx = max(0, min(idx, len(future)-1))
                row = future[idx]
            else:
                row = future[-1]
                row = dict(row)
                row["timestamp"] = target_noon.isoformat()
            row_classified = classify_pollutants(row["PM2_5"], row["PM10"], row["CO2"])
            date_pred = {
                "timestamp": row["timestamp"],
                "PM2_5": row["PM2_5"],
                "PM10": row["PM10"],
                "CO2": row["CO2"],
                "classified": row_classified
            }
        except Exception:
            date_pred = None

    # Choose dominant pollutant from CURRENT status for industry blame
    dom = find_dominant_pollutant(current_status)
    industry = pick_responsible_industry(lat, lon, dom.replace("_", "."))  # handle PM2_5→PM2.5 naming

    return jsonify({
        "city": city,
        "coords": {"lat": lat, "lon": lon},
        "history_72h": history,
        "current_status": current_status,
        "future_72h": future,
        "prediction_on_date": date_pred,
        "industry": industry
    })

# Optional anomaly endpoint using Isolation Forest (values = [PM2.5, PM10, CO2])
@app.route("/anomaly", methods=["POST"])
def anomaly():
    if iso_forest is None:
        return jsonify({"error": "IsolationForest not available"}), 503
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    vals = body.get("values")
    if not vals or len(vals) != 3:
        return jsonify({"error": "Send 'values' as [PM2.5, PM10, CO2]"}), 400
    pred = iso_forest.predict([vals])[0]  # -1 anomaly, 1 normal
    return jsonify({"anomaly": bool(pred == -1)})

if __name__ == "__main__":
    # Place index.html under ./templates/index.html
    app.run(debug=True)

