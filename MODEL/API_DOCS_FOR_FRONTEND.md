# ⚓ Intelligent Freight Forecasting API — Frontend Integration Guide

Welcome! This backend API provides machine-learning-powered freight rate forecasting and vessel chartering optimization for overseas bulk cargo procurement into the East Coast of India.

---

## 🚀 1. Quick Start (Run Backend Locally)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API Server
python server.py
# Or: uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Base API URL**: `http://127.0.0.1:8000`

---

## 📡 2. Core API Endpoints

### 1. `GET /api/forecast/90days`
Returns 90-day forward predictions for Baltic Dry Indices (Capesize, Panamax, Supramax, Handysize).
* **Response**:
```json
{
  "dates": ["2019-08-01", "2019-08-02", ...],
  "CI": [3657.5, 3656.5, ...],
  "PI": [1953.7, 1844.4, ...],
  "SI": [974.0, 963.7, ...],
  "HSI": [520.1, 520.2, ...]
}
```

---

### 2. `POST /api/optimizer/match-vessels`
Matches cargo parcels against physical vessels and East Coast port berth limits (Vizag, Paradip, Gangavaram, Dhamra, Gopalpur, Haldia).
* **Request Body**:
```json
{
  "origin": "Australia",
  "cargo_type": "Coking Coal",
  "tonnage": 150000,
  "destination_port": "Vizag"
}
```
* **Response**:
```json
{
  "status": "success",
  "recommendation": {
    "optimal_vessel_class": "Capesize",
    "voyages_required": 1,
    "parcel_per_voyage_mt": 150000,
    "turnaround_days_in_port": 3.3,
    "sea_transit_days": 18.0,
    "total_matching_vessels_in_dataset": 91,
    "sample_matching_vessels_from_dataset": [
      {
        "vessel_name": "MINERAL NINGBO",
        "deadweight_tonnage": 178000,
        "length_overall_m": 292.0,
        "beam_width_m": 45.0
      }
    ]
  }
}
```

---

### 3. `POST /api/models/{index_name}/predict-dynamic`
Runs real-time machine learning inference for a specific index (`CI`, `PI`, `SI`, `HSI`).
* **Request Body (Example using test sample index)**:
```json
{
  "vessel_index": "CI",
  "test_row_index": 5
}
```
* **Response**:
```json
{
  "vessel_index": "CI",
  "predicted_freight_index": 3382.43,
  "actual_ground_truth": 3230.0,
  "absolute_error": 152.43,
  "percentage_error": 4.72
}
```

---

### 4. `GET /api/models/{index_name}/feature-importance`
Returns top feature importance weights for rendering charts (`CI`, `PI`, `SI`, `HSI`).

---

## 💻 3. Frontend Integration Code (React / JavaScript)

```javascript
const API_BASE = 'http://127.0.0.1:8000';

// 1. Fetch 90-Day Forecast
export async function fetchForecast() {
  const res = await fetch(`${API_BASE}/api/forecast/90days`);
  return await res.json();
}

// 2. Optimize Vessel & Route
export async function optimizeVesselRoute(origin, cargo, tonnage, port) {
  const res = await fetch(`${API_BASE}/api/optimizer/match-vessels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      origin: origin,
      cargo_type: cargo,
      tonnage: Number(tonnage),
      destination_port: port
    })
  });
  return await res.json();
}
```
