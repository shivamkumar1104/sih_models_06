# Panamax Freight Forecasting Model (PI)

### 📊 Dataset Used:
- **Primary**: Baltic Panamax Index (dited BDI data1.xls) - 2,556 daily trading sessions (2012–2019)
- **Macro Features**: Global coal trade, US grain export cycles, crude bunker costs
- **Target Variable**: PI (Baltic Panamax Index)

### 🤖 Model Architecture:
- **Best Model**: **Gradient Boosting Regressor**
- **Evaluation Metrics on Out-of-Sample Test Data**:
  - **MAE**: 17.99 Index Points
  - **RMSE**: 30.80
  - **R² Score**: 0.9924 (99.24% accuracy)
  - **MAPE**: 1.29%

### 🚢 Vessel Segment Characteristics:
- **Typical Cargo**: Thermal coal (Indonesia/Australia), metallurgical coal, grains (US/Black Sea)
- **DWT Range**: 65,000 – 85,000 DWT
- **Port Compatibility**: Fully compatible with Vizag, Paradip, Gangavaram, Dhamra

### 💻 How to Load in Python:
`python
import joblib
model = joblib.load('PI_best_model.joblib')
`
