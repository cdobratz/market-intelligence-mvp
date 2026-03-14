"""
Train all ML models end-to-end.

Generates training data, trains XGBoost/RF/LightGBM regressors and classifiers,
evaluates with walk-forward validation, saves models, and logs to MLflow.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ---- Step 1: Generate training data ----
    logger.info("=" * 60)
    logger.info("STEP 1: Generating training data")
    logger.info("=" * 60)

    from src.data.sample_data_generator import generate_training_data

    data_dir = project_root / "data" / "processed"
    train_df, test_df = generate_training_data(
        output_dir=str(data_dir),
        n_samples=1000,
        n_symbols=5,
    )

    # Identify feature columns
    exclude_cols = [
        "date", "symbol", "target_return", "target_direction",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "direction_1d", "direction_5d", "direction_class_5d",
        "triple_barrier", "risk_adj_return_5d", "vol_20d",
    ]

    feature_cols = [
        c for c in train_df.columns
        if c not in exclude_cols
        and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Clean data
    for df in [train_df, test_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = train_df[feature_cols]
    y_train = train_df["target_return"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_return"]

    # Split train into train/val (80/20)
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_split = X_train.iloc[:-val_size]
    y_train_split = y_train.iloc[:-val_size]

    logger.info(f"Train: {X_train_split.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Features: {len(feature_cols)}")

    all_metrics = {}
    model_dir = project_root / "models"

    # ---- Step 2: Train XGBoost Regressor ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training XGBoost Regressor")
    logger.info("=" * 60)

    from src.models.supervised.xgboost_model import XGBoostRegressionModel

    xgb_model = XGBoostRegressionModel(
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        early_stopping_rounds=20,
    )
    xgb_model.train(X_train_split, y_train_split, X_val, y_val)
    xgb_train_metrics = xgb_model.evaluate(X_train_split, y_train_split, "train")
    xgb_test_metrics = xgb_model.evaluate(X_test, y_test, "test")
    xgb_model.save_model(str(model_dir / "xgboost_regressor"))
    all_metrics["xgboost"] = {**xgb_train_metrics, **xgb_test_metrics}

    # ---- Step 3: Train Random Forest Regressor ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Training Random Forest Regressor")
    logger.info("=" * 60)

    from src.models.supervised.random_forest_model import RandomForestRegressionModel

    rf_model = RandomForestRegressionModel(
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        }
    )
    rf_model.train(X_train_split, y_train_split)
    rf_train_metrics = rf_model.evaluate(X_train_split, y_train_split, "train")
    rf_test_metrics = rf_model.evaluate(X_test, y_test, "test")
    rf_model.save_model(str(model_dir / "random_forest_regressor"))
    all_metrics["random_forest"] = {**rf_train_metrics, **rf_test_metrics}

    # ---- Step 4: Train LightGBM Regressor ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training LightGBM Regressor")
    logger.info("=" * 60)

    from src.models.supervised.lightgbm_model import LightGBMRegressionModel

    lgbm_model = LightGBMRegressionModel(
        hyperparameters={
            "n_estimators": 200,
            "num_leaves": 31,
            "max_depth": 8,
            "learning_rate": 0.05,
        },
        early_stopping_rounds=20,
    )
    lgbm_model.train(X_train_split, y_train_split, X_val, y_val)
    lgbm_train_metrics = lgbm_model.evaluate(X_train_split, y_train_split, "train")
    lgbm_test_metrics = lgbm_model.evaluate(X_test, y_test, "test")
    lgbm_model.save_model(str(model_dir / "lightgbm_regressor"))
    all_metrics["lightgbm"] = {**lgbm_train_metrics, **lgbm_test_metrics}

    # ---- Step 5: Train Direction Classifiers ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Training Direction Classifiers")
    logger.info("=" * 60)

    from src.models.supervised.classification import DirectionClassifier

    # Create binary direction target (1 = up, 0 = down)
    y_train_cls = (y_train_split > 0).astype(int)
    y_val_cls = (y_val > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    # XGBoost Classifier
    logger.info("--- XGBoost Classifier ---")
    xgb_clf = DirectionClassifier(classifier_type="xgboost")
    xgb_clf.train(X_train_split, y_train_cls, X_val, y_val_cls)
    xgb_clf_metrics = xgb_clf.evaluate(X_test, y_test_cls, "test")
    xgb_clf.save_model(str(model_dir / "xgboost_classifier"))
    all_metrics["xgboost_classifier"] = xgb_clf_metrics

    # Random Forest Classifier
    logger.info("--- Random Forest Classifier ---")
    rf_clf = DirectionClassifier(classifier_type="random_forest")
    rf_clf.train(X_train_split, y_train_cls)
    rf_clf_metrics = rf_clf.evaluate(X_test, y_test_cls, "test")
    rf_clf.save_model(str(model_dir / "rf_classifier"))
    all_metrics["rf_classifier"] = rf_clf_metrics

    # ---- Step 6: Walk-Forward Validation ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Walk-Forward Validation (XGBoost)")
    logger.info("=" * 60)

    from src.models.supervised.regression import walk_forward_validation

    # Use full training data for walk-forward
    wf_results = walk_forward_validation(
        xgb_model, X_train, y_train, n_splits=5, test_size=50
    )
    all_metrics["walk_forward"] = {
        "mean_rmse": wf_results["mean_rmse"],
        "std_rmse": wf_results["std_rmse"],
        "mean_r2": wf_results["mean_r2"],
        "mean_directional_accuracy": wf_results["mean_directional_accuracy"],
    }

    # ---- Step 7: Feature Selection ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Feature Selection")
    logger.info("=" * 60)

    from src.features.selection import FeatureSelector

    selector = FeatureSelector(method="mutual_info")
    selected = selector.select_features(X_train, y_train, n_features=20)
    logger.info(f"Selected {len(selected)} features: {selected[:10]}...")
    all_metrics["feature_selection"] = {
        "n_selected": len(selected),
        "top_10": selected[:10],
    }

    # ---- Step 8: Save production model ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Selecting & saving production model")
    logger.info("=" * 60)

    # Find best model by test R2
    best_model_name = None
    best_r2 = -float("inf")
    for name in ["xgboost", "random_forest", "lightgbm"]:
        r2 = all_metrics[name].get("test_r2", -float("inf"))
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    logger.info(f"Best model: {best_model_name} (test R2: {best_r2:.6f})")

    # Copy best model to production dir
    import shutil
    production_dir = model_dir / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    source_dir = model_dir / f"{best_model_name}_regressor"
    for f in source_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, production_dir / f.name)

    # Save production metadata
    production_meta = {
        "model_name": best_model_name,
        "selected_at": datetime.now().isoformat(),
        "test_r2": best_r2,
        "test_rmse": all_metrics[best_model_name].get("test_rmse"),
        "test_directional_accuracy": all_metrics[best_model_name].get("test_directional_accuracy"),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    with open(production_dir / "production_metadata.json", "w") as f:
        json.dump(production_meta, f, indent=2)

    # ---- Step 9: Log to MLflow ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Logging to MLflow")
    logger.info("=" * 60)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("market-intelligence-production")

        with mlflow.start_run(run_name=f"production_{best_model_name}"):
            for key, value in all_metrics[best_model_name].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            mlflow.log_param("model_type", best_model_name)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("train_samples", len(X_train_split))
            mlflow.log_param("test_samples", len(X_test))

            # Log all model metrics as artifact
            with open("temp_all_metrics.json", "w") as f:
                json.dump(
                    {k: {mk: float(mv) if isinstance(mv, (int, float, np.floating)) else str(mv)
                         for mk, mv in v.items()} if isinstance(v, dict) else v
                     for k, v in all_metrics.items()},
                    f, indent=2, default=str,
                )
            mlflow.log_artifact("temp_all_metrics.json")
            os.unlink("temp_all_metrics.json")

        logger.info("MLflow logging complete")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    for name in ["xgboost", "random_forest", "lightgbm"]:
        m = all_metrics[name]
        logger.info(
            f"  {name:20s} | Train R2: {m.get('train_r2', 0):.4f} | "
            f"Test R2: {m.get('test_r2', 0):.4f} | "
            f"Test RMSE: {m.get('test_rmse', 0):.6f} | "
            f"Dir Acc: {m.get('test_directional_accuracy', 0):.2f}%"
        )

    for name in ["xgboost_classifier", "rf_classifier"]:
        m = all_metrics[name]
        logger.info(
            f"  {name:20s} | Accuracy: {m.get('test_accuracy', 0):.4f} | "
            f"F1: {m.get('test_f1', 0):.4f}"
        )

    logger.info(f"\n  Production model: {best_model_name} (R2: {best_r2:.4f})")
    logger.info(f"  Walk-forward R2: {all_metrics['walk_forward']['mean_r2']:.4f}")
    logger.info(f"  Selected features: {all_metrics['feature_selection']['n_selected']}")
    logger.info(f"\n  Models saved to: {model_dir}")
    logger.info(f"  Production model: {production_dir}")

    # Save summary
    summary_path = model_dir / "training_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "metrics": {
            k: {mk: float(mv) if isinstance(mv, (int, float, np.floating)) else str(mv)
                for mk, mv in v.items()} if isinstance(v, dict) else v
            for k, v in all_metrics.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
