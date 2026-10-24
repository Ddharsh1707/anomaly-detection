"""Anomaly Detection - finds faulty machine sensor readings with Isolation Forest."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def make_sensor_data(seed=0):
    """Normal temperature/vibration readings plus a few faulty ones."""
    rng = np.random.default_rng(seed)
    normal = np.column_stack([rng.normal(70, 3, 980), rng.normal(0.5, 0.05, 980)])
    faulty = np.column_stack([rng.uniform(85, 100, 20), rng.uniform(0.8, 1.2, 20)])
    data = np.vstack([normal, faulty])
    labels = np.r_[np.zeros(980), np.ones(20)].astype(int)
    order = rng.permutation(len(data))
    return pd.DataFrame(data[order], columns=["temperature_c", "vibration_g"]), labels[order]


def main():
    df, is_fault = make_sensor_data()
    print(f"{len(df)} readings, {is_fault.sum()} real faults (unknown to the model)\n")

    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=0)
    flagged = (model.fit_predict(df) == -1).astype(int)   # -1 means anomaly

    print(classification_report(is_fault, flagged, target_names=["normal", "fault"], digits=3))
    df["anomaly_score"] = -model.score_samples(df)
    print("Most suspicious readings:")
    print(df.sort_values("anomaly_score", ascending=False).head(5).round(3).to_string())

    plt.scatter(df.temperature_c, df.vibration_g, c=flagged, cmap="coolwarm", s=12)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Vibration (g)")
    plt.title("Red = flagged as anomaly")
    plt.savefig("anomalies.png", bbox_inches="tight")
    print("\nSaved anomalies.png")


if __name__ == "__main__":
    main()
