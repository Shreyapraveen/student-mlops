import os
import time
import json
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(csv_path=os.path.join("data", "student-mat.csv")):
    # UCI student performance dataset uses ; as separator [web:94]
    df = pd.read_csv(csv_path, sep=";")
    return df


def get_numeric_feature_cols(include_prior_grades: bool):
    # Numeric features we will use (exactly what API expects)
    base_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    ]
    if include_prior_grades:
        return base_cols + ["G1", "G2"]
    else:
        return base_cols  # no prior grades


def split_features_target(df, include_prior_grades=True):
    target_col = "G3"
    feature_cols = get_numeric_feature_cols(include_prior_grades)
    # Ensure columns exist in df
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor():
    # Only numeric features, so simple StandardScaler is enough
    preprocessor = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    return preprocessor


def train_and_evaluate_models(X, y, config, include_prior_grades=True):
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    n_estimators = config["model"]["n_estimators"]
    max_depth = config["model"]["max_depth"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()

    results = []

    # Baseline model
    baseline_model = DummyRegressor(strategy="mean")
    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", baseline_model)
    ])

    # Linear Regression
    lr_model = LinearRegression()
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", lr_model)
    ])

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ])

    models = [
        ("DummyRegressor", baseline_pipeline),
        ("LinearRegression", lr_pipeline),
        ("RandomForestRegressor", rf_pipeline),
    ]

    best_model_name = None
    best_pipeline = None
    best_rmse = float("inf")

    flag = "with_G1_G2" if include_prior_grades else "without_G1_G2"

    for name, pipeline in models:
      logging.info(f"Training {name} ({flag})...")
      with mlflow.start_run(run_name=f"{name}_{flag}", nested=True):
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(
            f"{name} ({flag}) -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}, "
            f"Train time: {train_time:.3f}s"
        )

        # Log params/metrics/tags to MLflow
        mlflow.log_param("include_prior_grades", include_prior_grades)
        mlflow.log_param("model_type", name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        if name == "RandomForestRegressor":
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("train_time", train_time)

        mlflow.set_tag("version", "v1")

        results.append({
            "model": name,
            "include_prior_grades": include_prior_grades,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_pipeline = pipeline
            best_model_name = f"{name}_{flag}"

    return best_model_name, best_pipeline, results


def main():
    logging.info("Loading config...")
    config = load_config()

    logging.info("Loading data...")
    df = load_data()

    all_results = []
    best_overall_rmse = float("inf")
    best_overall_pipeline = None
    best_overall_name = None
    mlflow.set_experiment("student_performance")
    # Scenario 1: with G1, G2
    X1, y1 = split_features_target(df, include_prior_grades=True)
    name1, pipe1, res1 = train_and_evaluate_models(X1, y1, config, include_prior_grades=True)
    all_results.extend(res1)

    if min(r["rmse"] for r in res1) < best_overall_rmse:
        best_overall_rmse = min(r["rmse"] for r in res1)
        best_overall_pipeline = pipe1
        best_overall_name = name1

    # Scenario 2: without G1, G2
    X2, y2 = split_features_target(df, include_prior_grades=False)
    name2, pipe2, res2 = train_and_evaluate_models(X2, y2, config, include_prior_grades=False)
    all_results.extend(res2)

    if min(r["rmse"] for r in res2) < best_overall_rmse:
        best_overall_rmse = min(r["rmse"] for r in res2)
        best_overall_pipeline = pipe2
        best_overall_name = name2

    logging.info(f"Best overall model: {best_overall_name} with RMSE: {best_overall_rmse:.3f}")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.joblib")
    joblib.dump(best_overall_pipeline, model_path)
    logging.info(f"Saved best pipeline to {model_path}")

    # Save summary results for later table in report
    results_path = os.path.join("models", "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Saved all results summary to {results_path}")


if __name__ == "__main__":
    main()