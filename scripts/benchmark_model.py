
# FICHIER : scripts/benchmark_model.py

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from app.persistence import load_model
from app.preprocessing import preprocess_data
from app.training import split_data

MODEL_PATH = Path("models/fraud_logistic_regression.joblib")
DATA_PATH = Path("data/creditcard.csv")
REPORT_PATH = Path("reports/benchmark_model.json")

BATCH_SIZE = 100
ITERATIONS = 100
WARMUP = 10


def main():
    print("=== Benchmark du modèle ===")

    # Charger le modèle de confiance déjà sauvegardé.
    model = load_model(MODEL_PATH)

    # Charger les données et reconstruire Validation.
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    _, X_val, _, _, _, _ = split_data(
        X, y, random_state=42
    )

    # Utiliser un lot fixe de 100 transactions.
    X_batch = X_val.iloc[:BATCH_SIZE].copy()

    process = psutil.Process()

    # Échauffement : éviter de mesurer les premiers appels.
    for _ in range(WARMUP):
        model.predict_proba(X_batch)

    # Mesurer la mémoire du processus avant benchmark.
    memory_before = process.memory_info().rss

    latencies_ms = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()

        model.predict_proba(X_batch)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies_ms.append(elapsed_ms)

    memory_after = process.memory_info().rss

    # Temps pour un lot de 100 transactions.
    mean_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    # Débit mesuré sur les lots exécutés.
    total_transactions = BATCH_SIZE * ITERATIONS
    total_seconds = sum(latencies_ms) / 1000

    throughput = total_transactions / total_seconds

    # Taille du modèle enregistré.
    model_size_mb = MODEL_PATH.stat().st_size / (1024**2)

    # Variation de mémoire du processus.
    memory_delta_mb = (
        memory_after - memory_before
    ) / (1024**2)

    results = {
        "batch_size": BATCH_SIZE,
        "iterations": ITERATIONS,
        "warmup": WARMUP,
        "mean_batch_latency_ms": mean_latency_ms,
        "p95_batch_latency_ms": p95_latency_ms,
        "throughput_transactions_per_second": throughput,
        "model_size_mb": model_size_mb,
        "memory_delta_mb": memory_delta_mb,
        "evaluation_set": "validation",
    }

    print("\n=== Résultats ===")

    for key, value in results.items():
        print(f"{key}: {value}")

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print(f"\nRapport enregistré : {REPORT_PATH}")
    print("Le jeu de test final n'a pas été utilisé.")


if __name__ == "__main__":
    main()
