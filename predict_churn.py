"""
predict_churn.py  — Exported inference function for Part 2
==========================================================
Loads the trained model pipeline (best model by PR-AUC) and exposes predict_churn().
Works for both Logistic Regression and XGBoost artifacts.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "results" / "models" / "churn_model_v1.joblib"
_artifact   = joblib.load(_MODEL_PATH)
_prep       = _artifact["preprocessor"]
_clf        = _artifact["model"]
_feat_cols  = _artifact["feature_cols"]
_all_names  = _artifact["all_feature_names"]
_is_tree    = hasattr(_clf, "feature_importances_")

_DEFAULTS = {
    "age": 40.0,
    "tenure_months": 24.0,
    "monthly_charges": 65.0,
    "total_charges": 1500.0,
    "avg_monthly_gb_used": 12.0,
    "num_support_tickets": 1.0,
    "avg_monthly_minutes": 280.0,
    "satisfaction_score": 5.5,
    "num_additional_services": 2.0,
    "days_since_contact": 30.0,
    "contract_type": "Month-to-month",
    "internet_service": "DSL",
    "payment_method": "Electronic check",
    "gender": "Male",
    "phone_service": "Yes",
}


def predict_churn(customer_data: dict) -> dict:
    """
    Accepts a dictionary of customer features. Returns:
    {
        "churn_probability": float,   # 0.0 to 1.0
        "risk_tier": str,             # "high", "medium", or "low"
        "top_risk_factors": list      # top 3 features driving this prediction
    }
    """
    row = {k: customer_data.get(k, v) for k, v in _DEFAULTS.items()}

    tenure   = float(row["tenure_months"])
    monthly  = float(row["monthly_charges"])
    tickets  = float(row["num_support_tickets"])
    sat      = float(row["satisfaction_score"])

    row["charge_tenure_ratio"] = monthly / (tenure + 1)
    row["support_burden"]      = tickets / (tenure + 1)
    row["sat_x_tickets"]       = (10 - sat) * tickets
    row["longterm_mtm"]        = int(
        str(row["contract_type"]).lower() == "month-to-month" and tenure > 24
    )

    X_in   = pd.DataFrame([row])[_feat_cols]
    X_proc = _prep.transform(X_in)
    prob   = float(_clf.predict_proba(X_proc)[0, 1])

    if prob >= 0.65:
        risk_tier = "high"
    elif prob >= 0.35:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    try:
        if _is_tree:
            import shap
            _explainer  = shap.TreeExplainer(_clf)
            shap_row    = _explainer.shap_values(X_proc)[0]
        else:
            # Linear: coefficient × standardised feature value
            shap_row = _clf.coef_[0] * X_proc[0]

        top_idx     = np.argsort(np.abs(shap_row))[::-1][:3]
        top_factors = [
            {
                "feature": _all_names[i],
                "shap_value": round(float(shap_row[i]), 4),
                "direction": "increases churn risk" if shap_row[i] > 0 else "decreases churn risk",
            }
            for i in top_idx
        ]
    except Exception:
        if _is_tree:
            imp = pd.Series(_clf.feature_importances_, index=_all_names)
        else:
            imp = pd.Series(np.abs(_clf.coef_[0]), index=_all_names)
        top3        = imp.nlargest(3).index.tolist()
        top_factors = [{"feature": f, "shap_value": None, "direction": "unknown"} for f in top3]

    return {
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "top_risk_factors": top_factors,
    }


if __name__ == "__main__":
    sample = {"tenure_months": 3, "satisfaction_score": 2.5,
              "contract_type": "Month-to-month", "num_support_tickets": 4}
    print(json.dumps(predict_churn(sample), indent=2))
