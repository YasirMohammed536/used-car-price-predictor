from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# Resolve asset tracking paths relative to this script directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'model', 'columns.json')

# 1. Load the frozen production Random Forest weights and feature schemas
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, 'r') as f:
    feature_columns = json.load(f)

# 2. Mathematically accurate dataset mapping dictionary (Cleaned & Calibrated)
BRAND_MODEL_MAP = {
    "bmw": [
        "3 Series 320d", "3 Series GT", "5 Series", "7 Series", "X1", "X3"
    ],
    "ford": [
        "Ecosport", "Ecosport 1.5 DV5 MT Titanium", "Ecosport 1.5 DV5 MT Titanium Optional",
        "Endeavour", "Figo", "Figo Diesel EXI", "Figo Diesel Titanium", "Figo Diesel ZXI"
    ],
    "honda": [
        "Amaze", "Amaze S i-Dtech", "Amaze S i-Vtech", "Brio", "Brio S MT", 
        "City", "City 1.5 S MT", "City 1.5 V MT", "City ZX", "Civic", "Jazz"
    ],
    "hyundai": [
        "Creta", "Creta 1.6 CRDi SX", "EON Era Plus", "Elantra", "Elite i20", "Eon", 
        "Fluidic Verna", "Grand i10", "Grand i10 AT Asta", "Grand i10 Nios", "Grand i10 Sportz", 
        "New i20", "Santro", "Santro Xing", "Verna", "Verna 1.6 SX", "Verna 1.6 SX VTVT", 
        "i10", "i10 Magna", "i10 Magna 1.2", "i10 Sportz", "i10 Sportz 1.2", 
        "i20", "i20 1.2 Magna", "i20 Active", "i20 Sportz 1.2"
    ],
    "isuzu": [
        "MU-X", "D-Max" # Corrected string matching anomaly
    ],
    "land rover": [
        "Discovery Sport", "Range Rover Evoque"
    ],
    "mahindra": [
        "Bolero", "Scorpio", "Thar", "XUV500", "XUV500 W6 2WD", "XUV500 W8 2WD", "Xylo"
    ],
    "maruti suzuki": [
        "800", "Alto", "Alto 800 LXI", "Alto K10 VXI", "Alto LXi", "Alto-800", "Alto-K10", 
        "Baleno", "Celerio", "Celerio VXI AT", "Eeco", "Ertiga", "Ertiga VDI", "Ertiga ZDI", 
        "Omni", "Ritz", "Ritz VDi", "Swift", "Swift Dzire", "Swift Dzire VDI", "Swift Dzire VDi", 
        "Swift Dzire VXI", "Swift VDI", "Swift VDI BSIV", "maruti-suzuki-dzire", "Wagon R", "Wagon R LXI", 
        "Wagon R LXI CNG", "Wagon R VXI", "Wagon-R", "Wagon-R-1-0", "Zen-Estilo"
    ],
    "mercedes-benz": [
        "C-Class", "CLA", "E-Class", "GL-Class", "GLA", "M-Class", "S-Class"
    ],
    "other": [
        "A4", "A4 2.0 TDI", "A4 2.0 TDI 177 Bhp Premium Plus", "A6", "Accord", "Alcazar", 
        "Aura", "Beat", "Brezza", "CRV", "Ciaz", "Compass", "Cruze", "Duster", 
        "Duster 110PS Diesel RxZ", "Grand Vitara", "Hector", "Hector Plus", "Ignis", 
        "KWID", "KWID RXT", "Kushaq", "Micra", "Other", "Q3", "Q7", "S-Cross", "S-Presso", 
        "Seltos", "Sonet", "Sx4", "TUV", "Terrano", "Triber", "Venue", "Vitara-Brezza", 
        "WRV", "XF", "XL6", "XUV 300", "XUV700", "Xcent", "Zest"
    ],
    "skoda": [
        "Octavia", "Rapid", "Superb", "Superb Elegance 1.8 TSI AT"
    ],
    "tata": [
        "Altroz", "Harrier", "Hexa", "Nano", "Nexon", "Safari", "Tiago", "Tigor"
    ],
    "toyota": [
        "Camry", "Corolla Altis", "Etios", "Etios Liva", "Fortuner", "Fortuner 3.0 Diesel", 
        "Fortuner 4x2 AT", "Fortuner 4x2 Manual", "Glanza", "Innova", "Innova Crysta", "Innova Hycross"
    ],
    "volkswagen": [
        "Polo", "Polo 1.2 MPI Highline", "Polo 1.5 TDI Highline", "Polo Diesel Comfortline 1.2L", 
        "Polo Petrol Comfortline 1.2L", "Vento", "Vento Diesel Highline", "VentoTest"
    ]
}

ALL_BRANDS = sorted(list(BRAND_MODEL_MAP.keys()))

@app.route('/', methods=['GET'])
def home():
    # Render default view passing clean layout dependencies
    return render_template('index.html', brands=ALL_BRANDS, brand_model_map=BRAND_MODEL_MAP)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs out of form fields
        year = int(request.form['year'])
        kilometers = float(request.form['kilometers'])
        transmission = int(request.form['transmission'])
        owner_type = int(request.form['owner_type'])
        
        selected_brand = request.form['brand']
        selected_model = request.form['model']
        selected_fuel = request.form['fuel']

        # Construct vector structure tracking 211 parameters
        input_vector = np.zeros(len(feature_columns))
        
        # Insert base values at target coordinates
        input_vector[feature_columns.index('Year')] = year
        input_vector[feature_columns.index('Kilometers_Driven')] = kilometers
        input_vector[feature_columns.index('Transmission')] = transmission
        input_vector[feature_columns.index('Owner_Type')] = owner_type
        
        # Locate OHE coordinate paths and toggle hot state flag inputs to 1
        brand_col = f"Brand_{selected_brand}"
        if brand_col in feature_columns:
            input_vector[feature_columns.index(brand_col)] = 1
            
        fuel_col = f"Fuel_Type_{selected_fuel.lower()}"
        if fuel_col in feature_columns:
            input_vector[feature_columns.index(fuel_col)] = 1
            
        model_col = f"Model_encoded_{selected_model}"
        if model_col in feature_columns:
            input_vector[feature_columns.index(model_col)] = 1
            
        # Run inference computation array across production Random Forest trees
        raw_prediction = model.predict([input_vector])[0]
        final_valuation = f"₹ {round(raw_prediction, 2)} Lakhs"
        
        return render_template('index.html', 
                               prediction_text=final_valuation,
                               brands=ALL_BRANDS, 
                               brand_model_map=BRAND_MODEL_MAP,
                               selected_brand=selected_brand,
                               selected_model=selected_model)
                               
    except Exception as e:
        # Fail gracefully back to template layer showing exact index tracking issues
        return render_template('index.html', 
                               prediction_text=f"Error evaluating metrics: {str(e)}",
                               brands=ALL_BRANDS, 
                               brand_model_map=BRAND_MODEL_MAP)

if __name__ == '__main__':
    # Launch application on secure target port 8080
    app.run(debug=True, port=8080)