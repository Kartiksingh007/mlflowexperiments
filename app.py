

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(40)

    remote_server_uri="https://dagshub.com/kartiksingh007/mlflowexperiments.mlflow"
    mlflow.set_tracking_uri("file:./mlruns")
    # mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("wine-quality-elasticnet")


    CSV_URL = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master"
        "/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(CSV_URL, sep=";")
        print(f"Dataset loaded — {len(data):,} rows, {data.shape[1]} columns")
    except Exception as e:
        logger.exception("Failed to load dataset: %s", e)
        sys.exit(1)

 
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop("quality", axis=1)
    test_x  = test.drop("quality", axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]


    alpha    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

  
        preds = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        print(f"\nElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  R²   : {r2:.4f}")

        # ── Log PARAMS ───────────────────────────────────────
        mlflow.log_params({
            "alpha":    alpha,
            "l1_ratio": l1_ratio,
        })

        # ── Log METRICS ──────────────────────────────────────
        mlflow.log_metrics({
            "rmse": round(rmse, 4),
            "mae":  round(mae,  4),
            "r2":   round(r2,   4),
        })

        # ── Log TAGS ─────────────────────────────────────────
        mlflow.set_tags({
            "model_type": "ElasticNet",
            "dataset":    "winequality-red",
            "developer":  os.getenv("USER", "unknown"),
            "framework":  "scikit-learn",
        })

        # ── Log DATASET (shows under Inputs tab in UI) ───────
        dataset = mlflow.data.from_pandas(
            train,
            source=CSV_URL,
            name="winequality-red",
            targets="quality"
        )
        mlflow.log_input(dataset, context="training")

        # ── Build signature + input example ──────────────────
        signature     = infer_signature(train_x, model.predict(train_x))
        input_example = train_x.iloc[:5]

        # ── Log ARTIFACT (model) ─────────────────────────────
        tracking_uri_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_uri_scheme not in ("file", "mlruns", ""):
            mlflow.sklearn.log_model(
                sk_model              = model,
                artifact_path         = "model",
                signature             = signature,
                input_example         = input_example,
                registered_model_name = "ElasticnetWineModel",
            )
            print("Model registered in remote Model Registry")
        else:
            mlflow.sklearn.log_model(
                sk_model      = model,
                artifact_path = "model",
                signature     = signature,
                input_example = input_example,
            )
            print("Model artifact saved locally")
        run_id = mlflow.active_run().info.run_id
        print(f"\n  Run ID       : {run_id}")
        print(f"  Experiment   : wine-quality-elasticnet")
        print(f"  Tracking URI : {mlflow.get_tracking_uri()}")
        print("\n──────────────────────────────────────────")
        print("  To view UI → open terminal and run:")
        print("  mlflow ui --port 5000")
        print("  Then open → http://127.0.0.1:5000")
    