"""
ml_service.py

Clean ML pipeline for:
- Crop prediction (CatBoost)
- Fertilizer recommendation (LightGBM, 8 features)
- Irrigation advice (rule-based)
- Geo validation (using geo-referenced crop dataset)

This file is meant to be used by the backend/frontend team.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

# ==============================
# 0. CONFIG – CHANGE PATHS HERE
# ==============================

# Main crop + fertilizer dataset
DATA_PATH = "/Users/sayenaqureshi/Main-Folder/notebook/datasets/Crop and fertilizer dataset (2).csv"

# Geo-referenced crop dataset
GEO_PATH = "/Users/sayenaqureshi/Main-Folder/notebook/datasets/CropDataset-Enhanced.csv"

# Column names in GEO dataset
GEO_DISTRICT_COL = "Address"   # e.g. "Address"
GEO_CROPS_COL    = "Crop"      # e.g. "Crop"


# ===========================
# 1. GLOBALS (MODELS, LABELERS)
# ===========================

# Will be initialized in initialize_pipeline()
DF_MAIN = None
DF_GEO = None

LE_CROP = None
LE_FERT = None
LE_DIST_FERT = None
LE_SOIL_FERT = None

CROP_MODEL = None         # CatBoost model
FERT_MODEL_LGB = None     # LightGBM model (fertilizer, 8 features)

FEATURE_COLS = [
    "Nitrogen", "Phosphorus", "Potassium", "pH",
    "Rainfall", "Temperature", "District_Name", "Soil_color"
]

CAT_FEATURE_IDX = [
    FEATURE_COLS.index("District_Name"),
    FEATURE_COLS.index("Soil_color")
]


# ===========================
# 2. GEO VALIDATION UTILITIES
# ===========================

def _prepare_geo_dataset(geo_path: str) -> pd.DataFrame:
    """Load and clean the geo-referenced crop dataset."""
    df_geo = pd.read_csv(geo_path)

    df_geo["District_clean"] = (
        df_geo[GEO_DISTRICT_COL]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df_geo["Crops_clean"] = (
        df_geo[GEO_CROPS_COL]
        .astype(str)
        .str.lower()
    )

    return df_geo


def geo_validate_crop(district_name: str, predicted_crop: str, df_geo: pd.DataFrame):
    """
    Returns (status, message) where:
    - status ∈ {"supported", "unsupported", "unknown"}
    - message is a human-readable explanation
    """
    if district_name is None or predicted_crop is None:
        return "unknown", "No district or crop information available."

    d = str(district_name).lower().strip()
    c = str(predicted_crop).lower().strip()

    matches = df_geo[df_geo["District_clean"].str.contains(d, na=False)]
    if matches.empty:
        return "unknown", f"No geo data available for district '{district_name}'."

    all_crops_text = " , ".join(matches["Crops_clean"].astype(str).tolist())
    crop_list = [x.strip() for x in all_crops_text.split(",") if x.strip()]

    if c in crop_list:
        return "supported", (
            f"The predicted crop '{predicted_crop}' is commonly grown in "
            f"district '{district_name}' according to geo-referenced data."
        )
    else:
        return "unsupported", (
            f"Warning: The predicted crop '{predicted_crop}' is NOT commonly grown in "
            f"district '{district_name}' in geo-referenced data. "
            f"Please consult local experts or consider region-specific constraints."
        )


# ===========================
# 3. CORE MODEL UTILITIES
# ===========================

def train_catboost_model(X_train, X_test, y_train, y_test, cat_features_idx):
    """
    Train a CatBoost multi-class classifier with early stopping
    and return the trained model.
    """
    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
    valid_pool = Pool(X_test, y_test, cat_features=cat_features_idx)

    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=4,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=50,
    )
    return model


def irrigation_recommender(crop: str, rainfall: float, temp: float, soil_color: str) -> str:
    """Simple rule-based irrigation logic."""
    high_water_crops = ["Paddy", "Sugarcane", "Grape"]
    low_water_crops = ["Bajra", "Jowar", "Cotton"]

    # Base need by rainfall
    if rainfall < 600:
        base_need = "Heavy"
    elif rainfall < 1200:
        base_need = "Moderate"
    else:
        base_need = "Light"

    # Adjust by crop, soil and temperature
    if crop in high_water_crops and base_need in ["Moderate", "Light"]:
        schedule = "Daily Monitoring (High Volume)"
    elif soil_color in ["Red", "Sandy"] and temp > 30:
        schedule = "Daily Check (Medium Volume)"
    elif crop in low_water_crops and rainfall < 1000:
        schedule = "Weekly Check (Low Volume)"
    else:
        schedule = "Every 2-3 Days"

    return f"{base_need} Irrigation ({schedule})"


# ===========================
# 4. PIPELINE INITIALIZATION
# ===========================

def initialize_pipeline(
    data_path: str = DATA_PATH,
    geo_path: str = GEO_PATH,
):
    """
    Load data, train models, and initialize global variables.
    This should be called once at startup (or will be called lazily).
    """
    global DF_MAIN, DF_GEO
    global LE_CROP, LE_FERT, LE_DIST_FERT, LE_SOIL_FERT
    global CROP_MODEL, FERT_MODEL_LGB

    # --- Load datasets ---
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    DF_MAIN = df

    df_geo = _prepare_geo_dataset(geo_path)
    DF_GEO = df_geo

    # --- Features & targets ---
    X = df[FEATURE_COLS].copy()
    y_crop = df["Crop"].copy()
    y_fert = df["Fertilizer"].copy()

    # Ensure categoricals as string for CatBoost
    X["District_Name"] = X["District_Name"].astype(str)
    X["Soil_color"] = X["Soil_color"].astype(str)

    # --- Encode targets ---
    LE_CROP = LabelEncoder()
    LE_FERT = LabelEncoder()

    y_crop_le = LE_CROP.fit_transform(y_crop)
    y_fert_le = LE_FERT.fit_transform(y_fert)

    # --- Train/test split for crop (CatBoost) ---
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_crop_le,
        test_size=0.2,
        random_state=42,
        stratify=y_crop_le,
    )

    # Train CatBoost crop model
    CROP_MODEL = train_catboost_model(
        X_train_c, X_test_c, y_train_c, y_test_c, CAT_FEATURE_IDX
    )

    # --- Train LightGBM for Fertilizer using 8 features ---
    # Reuse the same 8 feature columns
    X_fert_8 = df[FEATURE_COLS].copy()
    y_fert_labels = df["Fertilizer"].copy()

    LE_DIST_FERT = LabelEncoder()
    LE_SOIL_FERT = LabelEncoder()

    X_fert_8_enc = X_fert_8.copy()
    X_fert_8_enc["District_Name"] = LE_DIST_FERT.fit_transform(
        X_fert_8_enc["District_Name"].astype(str)
    )
    X_fert_8_enc["Soil_color"] = LE_SOIL_FERT.fit_transform(
        X_fert_8_enc["Soil_color"].astype(str)
    )

    y_fert_enc = LE_FERT.transform(y_fert_labels)

    X_train_f8, X_test_f8, y_train_f8, y_test_f8 = train_test_split(
        X_fert_8_enc, y_fert_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_fert_enc,
    )

    fert_model_lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
    )

    fert_model_lgb.fit(X_train_f8, y_train_f8)
    FERT_MODEL_LGB = fert_model_lgb

    print("[ml_service] Pipeline initialized successfully.")
    print("[ml_service] DF_MAIN shape:", DF_MAIN.shape)
    print("[ml_service] DF_GEO shape :", DF_GEO.shape)


def _ensure_initialized():
    """Lazy guard so that get_full_recommendation can be called safely."""
    if any(x is None for x in [DF_MAIN, DF_GEO, LE_CROP, LE_FERT, LE_DIST_FERT,
                               LE_SOIL_FERT, CROP_MODEL, FERT_MODEL_LGB]):
        initialize_pipeline()


# ===========================
# 5. INFERENCE API
# ===========================

def get_full_recommendation(input_dict: dict) -> dict:
    """
    Main function to be called by the backend.

    Parameters
    ----------
    input_dict : dict
        {
          "Nitrogen": float,
          "Phosphorus": float,
          "Potassium": float,
          "pH": float,
          "Rainfall": float,
          "Temperature": float,
          "District_Name": "string",
          "Soil_color": "string"
        }

    Returns
    -------
    dict
        {
          "crop": str,
          "fertilizer": str,
          "irrigation": str,
          "link": str,
          "geo_status": "supported"/"unsupported"/"unknown",
          "geo_message": str
        }
    """
    _ensure_initialized()

    # --- Convert to DataFrame ---
    user_df = pd.DataFrame([input_dict])

    # --- 1) Crop prediction – CatBoost ---
    user_df_cat = user_df.copy()
    user_df_cat["District_Name"] = user_df_cat["District_Name"].astype(str)
    user_df_cat["Soil_color"] = user_df_cat["Soil_color"].astype(str)

    pred_crop_le = CROP_MODEL.predict(user_df_cat).flatten().astype(int)[0]
    pred_crop_label = LE_CROP.inverse_transform([pred_crop_le])[0]

    # --- 2) Fertilizer prediction – LightGBM with encoded categoricals ---
    user_df_enc = user_df.copy()
    user_df_enc["District_Name"] = LE_DIST_FERT.transform(
        [str(user_df_enc["District_Name"].iloc[0])]
    )[0]
    user_df_enc["Soil_color"] = LE_SOIL_FERT.transform(
        [str(user_df_enc["Soil_color"].iloc[0])]
    )[0]

    pred_fert_enc = FERT_MODEL_LGB.predict(user_df_enc)[0]
    pred_fert_label = LE_FERT.inverse_transform([pred_fert_enc])[0]

    # --- 3) Irrigation advice ---
    irrigation_advice = irrigation_recommender(
        pred_crop_label,
        user_df["Rainfall"].iloc[0],
        user_df["Temperature"].iloc[0],
        user_df["Soil_color"].iloc[0],
    )

    # --- 4) Educational Link lookup ---
    link_row = DF_MAIN[
        (DF_MAIN["Crop"] == pred_crop_label)
        & (DF_MAIN["Fertilizer"] == pred_fert_label)
    ]

    if not link_row.empty and "Link" in link_row.columns:
        link = link_row["Link"].iloc[0]
    else:
        link = "No link available for this combination."

    # --- 5) Geo validation ---
    geo_status, geo_message = geo_validate_crop(
        user_df["District_Name"].iloc[0],
        pred_crop_label,
        DF_GEO,
    )

    return {
        "crop": pred_crop_label,
        "fertilizer": pred_fert_label,
        "irrigation": irrigation_advice,
        "link": link,
        "geo_status": geo_status,
        "geo_message": geo_message,
    }


# ===========================
# 6. OPTIONAL: LOCAL TEST
# ===========================

if __name__ == "__main__":
    # Simple manual test – you can run: python ml_service.py
    initialize_pipeline()

    sample_input = {
        "Nitrogen": 150.0,
        "Phosphorus": 68.0,
        "Potassium": 68.0,
        "pH": 4.5,
        "Rainfall": 900.0,
        "Temperature": 30.0,
        "District_Name": "Satara",
        "Soil_color": "Black",
    }

    result = get_full_recommendation(sample_input)
    print("\n=== Test Inference ===")
    print("Input:", sample_input)
    print("Output:", result)
