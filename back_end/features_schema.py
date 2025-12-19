"""
Shared feature schema for TunnelVision
-----------------------------------
This file is the single source of truth for:
- feature names
- feature order (CRITICAL)
- expected types / notes

Both training code and the API should import and use this file
so features never drift.
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Feature order (DO NOT CHANGE without retraining all models)
# ---------------------------------------------------------------------------

FEATURE_ORDER: List[str] = [
    "type",                    # encoded categorical
    "material",                # encoded categorical
    "region",                  # encoded categorical / fallback location
    "soil_type",               # encoded categorical
    "latitude",                # float
    "longitude",               # float
    "temperature_c",           # float (inferred)
    "last_repair_date",        # str
    "snapshot_date",           # str
    "install_year",            # int
    "length_m"                 # float        # encoded / embedded
]

# ---------------------------------------------------------------------------
# Optional: human-readable index map (great for debugging)
# ---------------------------------------------------------------------------

FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_ORDER)}

# ---------------------------------------------------------------------------
# Helper: build a feature vector in the correct order
# ---------------------------------------------------------------------------

def build_feature_vector(values: Dict[str, Any]) -> List[Any]:
    """
    Build a feature vector in the correct order.

    Example:
        features = build_feature_vector({
            "type": "Road",
            "material": "Asphalt",
            "region": "Santa Clara",
            "soil_type": "Gravel",
            "latitude": 37.33,
            "longitude": -121.89,
            "temperature_c": 18.7,
            "last_repair_date": "01-01-2020",
            "snapshot_date": "01-01-2024",
            "install_year": 2012,
            "length_m": 150.47
        })
    """

    missing = [f for f in FEATURE_ORDER if f not in values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return [values[name] for name in FEATURE_ORDER]

# ---------------------------------------------------------------------------
# Helper: quick schema assertion (use in train.py or API startup)
# ---------------------------------------------------------------------------

def assert_feature_length(features: List[Any]) -> None:
    if len(features) != len(FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(FEATURE_ORDER)} features, got {len(features)}"
        )
