import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .train import load_data, load_config, split_features_target, get_numeric_feature_cols


def main():
    config = load_config()
    df = load_data()

    # For now, we evaluate using include_prior_grades=True
    include_prior_grades = True
    X, y = split_features_target(df, include_prior_grades=include_prior_grades)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    model_path = os.path.join("models", "model.joblib")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation with saved pipeline -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

    # Residual plot
    residuals = y_test - y_pred
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted G3")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    residual_path = os.path.join("models", "residuals.png")
    plt.savefig(residual_path)
    print(f"Saved residual plot to {residual_path}")

    # Feature importance (only if final model is RandomForest)
    try:
        # Pipeline: preprocessor -> model
        model = pipeline.named_steps["model"]
        feature_cols = get_numeric_feature_cols(include_prior_grades)
        importances = model.feature_importances_
        if len(importances) == len(feature_cols):
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(importances)), importances, tick_label=feature_cols)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Importance")
            plt.title("Feature Importance (RandomForest)")
            plt.tight_layout()
            fi_path = os.path.join("models", "feature_importance.png")
            plt.savefig(fi_path)
            print(f"Saved feature importance plot to {fi_path}")
        else:
            print("Feature importance length does not match feature columns; skipping plot.")
    except Exception as e:
        print(f"Could not compute feature importance (likely non-RF model): {e}")


if __name__ == "__main__":
    main()