# Ship & Vessel Classification Engine

### 📊 Dataset Used:
- **File**: Cleaned_ships_data.csv (200 real bulk carrier vessel specifications)
- **Attributes**: Vessel Class, Deadweight Tonnage (DWT), Gross Tonnage (GT), Length Overall (LOA), Beam (Width)

### 📐 Vessel Category Boundaries:
| Vessel Class | Count | Avg DWT (MT) | Avg Length (m) | Avg Width (m) |
| :--- | :---: | :---: | :---: | :---: |
| **Capesize** | 158 | 209,758 | 325.1 m | 51.5 m |
| **Panamax** | 31 | 86,424 | 293.6 m | 46.0 m |
| **Supramax** | 1 | 55,738 | 296.0 m | 38.0 m |
| **Handysize** | 10 | 11,753 | 307.2 m | 39.8 m |

### 🎯 Purpose in Project:
Translates cargo parcel requirements (e.g. 150,000 MT coking coal) into physical vessel recommendations while verifying dimensions against port berthing limits.
