# Used Car Price Predictor

![Application Interface](screenshot.png)

An end-to-end machine learning project that predicts used car resale 
prices in the Indian market. Includes a locally deployable Flask web 
application with a stateful frontend interface built with Tailwind CSS.

---

## Project Overview

Built to demonstrate a complete ML pipeline from raw data sourcing 
through preprocessing, model comparison, hyperparameter tuning, and 
local web deployment. Two independently sourced Kaggle datasets were 
merged into a single training set after resolving significant 
incompatibility and formatting issues.

---

## Model Performance

| Model | Test R² | RMSE (Lakhs) | Train R² |
|---|---|---|---|
| Linear Regression | 0.67 | 6.57 | 0.70 |
| Decision Tree (depth=10) | 0.61 | 7.11 | 0.78 |
| SVR (RBF kernel) | 0.54 | 7.73 | 0.56 |
| Random Forest (baseline) | 0.96 | 5.90 | 0.96 |
| Random Forest (tuned) | 0.74 | 5.86 | 0.93 |

Final model: Random Forest Regressor tuned via RandomizedSearchCV  
(`n_estimators=300, max_depth=None, min_samples_split=5, 
min_samples_leaf=1`).

**Why the tuned model was chosen over the baseline:**
The baseline Random Forest showed identical train and test R² of 0.96, 
which is unusual and indicates the cross-validated score was optimistic. 
RandomizedSearchCV with 3-fold cross-validation produced a more honest 
generalisation estimate of 0.74. The tuned model was selected for its 
trustworthiness over its raw metric.

**Honest limitations:**
RMSE of 5.86 Lakhs against a dataset median price of 5 Lakhs means 
predictions carry meaningful error. The model performs most reliably 
on high-volume mass-market vehicles — Maruti Suzuki, Hyundai, Honda — 
which dominate the training data. Predictions on luxury outliers 
(Mercedes-Benz, Audi, BMW, Porsche) are less reliable due to limited 
training samples in that price segment.

---

## Data Sources

Both datasets sourced from Kaggle:
- `used_cars_data.csv` — Indian used car listings with detailed 
  vehicle specifications
- `used_car_dataset.csv` — Secondary Indian used car listings

Raw files included in `data/` for full reproducibility.

**Final training set:** 14,222 rows, 211 features after preprocessing.

---

## Data Engineering & Preprocessing Challenges

Preprocessing was the most time-consuming phase of this project. 
Both datasets had incompatible schemas and required significant 
cleaning before a usable training set could be produced.

**Schema incompatibility across datasets**
Dataset one stored vehicle identity as a single combined Name string 
requiring splitting into separate Brand and Model fields using 
`str.split(n=1)`. Dataset two had separate Brand and model columns 
but used different column names, requiring renaming and reordering 
before the datasets could be merged via `pd.concat`.

**Inconsistent data formatting**
Price was stored differently across both datasets. Dataset one used 
numeric Lakh values. Dataset two stored raw Rupee strings 
(e.g. ₹1,95,000) requiring ₹ symbol stripping, comma removal, 
type conversion, and division by 100,000 to standardise to Lakhs. 
Kilometer values had mixed formats — comma-separated numbers, 
unit suffixes (" km"), and decimal inconsistencies — requiring 
multi-step string cleaning before numeric conversion.

**Brand name inconsistency**
One manufacturer appeared under different name variants across the 
two datasets. All variants were standardised to a single consistent 
label to prevent the model treating the same brand as separate 
categories.

**Model column high cardinality**
The Model column contained 1,872 unique values — full specification 
strings (e.g. "Creta 1.6 CRDi SX Option") rather than clean model 
names. Initially dropped due to encoding complexity. Later 
reintroduced by grouping models with fewer than 15 listings into an 
Other category using a frequency threshold filter. This change 
improved test R² from 0.72 to 0.74 and reduced RMSE from 6.12 to 
5.86 Lakhs — a measurable improvement from a single feature 
engineering decision.

**Outlier and missing data handling**
- Price floor applied at 0.5 Lakhs to remove data entry errors
- Kilometer cap applied at 300,000 to remove unrealistic values
- Cars with under 500 km driven manufactured before 2022 removed 
  as implausible listings
- Null Kilometers_Driven values imputed with column median
- Rare brands below 10 listings grouped into Other
- Rare fuel types below 21 listings grouped into Other

**Deduplication**
After merging both datasets a significant portion of duplicate rows 
were identified and removed. The two source datasets shared listings 
from the same underlying market data.

---

## Feature Engineering

- Brand and Fuel_Type one-hot encoded via `pd.get_dummies`
- Model_encoded frequency-threshold grouped then one-hot encoded
- Owner_Type ordinal encoded: fourth & above=0, third=1, 
  second=2, first=3
- Transmission label encoded: automatic=0, manual=1
- Final feature matrix: 211 columns

---

## Tech Stack

- **Data & Modeling:** Python, Pandas, NumPy, Scikit-Learn
- **Web Framework:** Flask with Jinja2 templating
- **Frontend:** HTML5, Tailwind CSS v4, JavaScript
- **Version Control:** Git, Git LFS (model file: 120.27MB)

---

## Web Application Features

**Stateful form interface**
User inputs persist after form submission. Selected brand, model, 
year, kilometers, transmission, fuel type, and ownership history 
maintain their values rather than resetting on prediction request.

**Dependent dropdown filtering**
Model dropdown dynamically filters based on selected brand using 
JavaScript DOM manipulation. Only models available for the selected 
brand are shown, preventing users from submitting brand/model 
combinations that don't exist in the training data — protecting the 
model from out-of-distribution inputs.

**Multi-word brand handling**
Custom string matching in the Flask backend handles multi-word brand 
names (e.g. Maruti Suzuki, Land Rover, Mercedes-Benz) correctly 
when constructing the one-hot encoded feature vector.

---

## How to Run Locally

```bash
git clone https://github.com/YasirMohammed536/used-car-price-predictor
cd used-car-price-predictor
pip install -r requirements.txt
python app/app.py
```

Open browser at `http://127.0.0.1:8080`

---

## What I Would Do Differently

- Extract clean base model names from the spec strings using NLP 
  rather than frequency threshold grouping
- Retain Engine, Power, and Mileage features from dataset one 
  rather than dropping them to enable the merge — these are 
  meaningful price predictors that were sacrificed for compatibility
- Collect fresher data via direct scraping from OLX and CarDekho 
  for more current market prices and broader coverage
- Apply log transformation to Price during training to reduce RMSE 
  on high-value vehicles

---

## Author

Yasir Mohammed  
Self-directed ML learner  
GitHub: YasirMohammed536
