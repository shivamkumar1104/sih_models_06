# Capesize Freight Forecasting Model (CI)

### 📊 Dataset Used:
- **Primary**: Baltic Capesize Index (dited BDI data1.xls) - 2,556 daily trading sessions (2012–2019)
- **Macro Features**: Global iron ore supply (Australia/Brazil to Asia), met coal from US/Mozambique/Russia
- **Target Variable**: CI (Baltic Capesize Index)

### 🤖 Model Architecture:
- **Best Model**: **Gradient Boosting Regressor**
- **Evaluation Metrics on Out-of-Sample Test Data**:
  - **MAE**: 58.51 Index Points
  - **RMSE**: 79.61
  - **R² Score**: 0.9942 (99.42% accuracy)
  - **MAPE**: 7.79%

### 🚢 Vessel Segment Characteristics:
- **Typical Cargo**: High-volume Iron Ore & Heavy Coking Coal parcels (120k–180k MT)
- **DWT Range**: 100,000 – 220,000 DWT
- **Port Compatibility**: Deep-water berths at Vizag Outer Harbor, Gangavaram (21m draft), Dhamra (18m draft)

### 💻 How to Load in Python:
`python
import joblib
model = joblib.load('CI_best_model.joblib')
`
