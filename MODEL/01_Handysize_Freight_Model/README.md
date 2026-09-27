# Handysize Freight Forecasting Model (HSI)

### 📊 Dataset Used:
- **Primary**: Baltic Handysize Index (dited BDI data1.xls) - 2,556 daily trading sessions (2012–2019)
- **Macro Features**: 28 Global Commodity Prices (Crude, Coal, Metals) & US Seaborne Trade data
- **Target Variable**: HSI (Baltic Handysize Index)

### 🤖 Model Architecture:
- **Best Model**: **Gradient Boosting Regressor**
- **Evaluation Metrics on Out-of-Sample Test Data**:
  - **MAE**: 2.60 Index Points
  - **RMSE**: 3.63
  - **R² Score**: 0.9988 (99.88% accuracy)
  - **MAPE**: 0.58%

### 🚢 Vessel Segment Characteristics:
- **Typical Cargo**: Minor bulks, steel, fertilizers, agri-commodities, bauxite
- **DWT Range**: 10,000 – 35,000 DWT
- **Port Flexibility**: Suitable for shallow draft ports like Haldia and Gopalpur

### 💻 How to Load in Python:
`python
import joblib
model = joblib.load('HSI_best_model.joblib')
`
