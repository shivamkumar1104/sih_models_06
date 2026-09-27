# East Coast India Port Optimization Engine

### 📊 Datasets Used:
- TableNo11_TRAFFIC_HANDLED_AT_VISHAKHAPATNAM_PORT.csv: 71 years of historical port throughput data
- 1640237145_Berth Details.pdf: Port official berth dimensions, permissible drafts, and handling rates

### ⚓ Port Constraint Reference Table:
| Port Name | Max Draft (m) | Max LOA (m) | Max Beam (m) | Discharge Rate (TPD) | Permissible Vessels |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Vizag** | 18.1 | 350 | 55 | 45,000 MT/day | Handysize, Supramax, Panamax, Capesize |
| **Gangavaram** | 21.0 | 330 | 57 | 50,000 MT/day | Handysize, Supramax, Panamax, Capesize |
| **Dhamra** | 18.0 | 330 | 55 | 40,000 MT/day | Handysize, Supramax, Panamax, Capesize |
| **Paradip** | 14.5 | 300 | 50 | 35,000 MT/day | Handysize, Supramax, Panamax (Partial Capesize) |
| **Gopalpur** | 14.0 | 230 | 38 | 15,000 MT/day | Handysize, Supramax |
| **Haldia** | 9.5 | 200 | 32 | 12,000 MT/day | Handysize Only (Draft Restricted) |
| **Sagar/Sandheads** | 9.0 | 200 | 30 | 10,000 MT/day | Lighterage & Transshipment |

### 🎯 Optimization Functionality:
Calculates berth feasibility, turnaround days, demurrage risk, and recommended discharge port given vessel size and parcel tonnage.
