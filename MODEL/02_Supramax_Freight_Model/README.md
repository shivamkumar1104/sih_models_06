# Supramax Freight Forecasting Model (SI)

### 📊 Dataset Used:
- **Primary**: Baltic Supramax Index (dited BDI data1.xls) - 2,556 daily trading sessions (2012–2019)
- **Macro Features**: Commodity supply chain data + Ocean services cost indices
- **Target Variable**: SI (Baltic Supramax Index)

### 🤖 Model Architecture:
- **Best Model**: **Ridge Regression (L2 Regularization)**
- **Evaluation Metrics on Out-of-Sample Test Data**:
  - **MAE**: 7.17 Index Points
  - **RMSE**: 9.13
  - **R² Score**: 0.9976 (99.76% accuracy)
  - **MAPE**: 0.95%

### 🚢 Vessel Segment Characteristics:
- **Typical Cargo**: Grain, coal, nickel ore, fertilizers, cement clinker
- **DWT Range**: 45,000 – 60,000 DWT (equipped with onboard cranes/grabs)
- **Port Flexibility**: High - ideal for medium East Coast ports (Paradip, Gopalpur)

### 💻 How to Load in Python:
`python
import joblib
model = joblib.load('SI_best_model.joblib')
`
