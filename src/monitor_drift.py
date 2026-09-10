import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_data_drift():
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)

    drift_report.save_html("drift_report.html")
    report_dict = drift_report.as_dict()

    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset Drift Detected: {dataset_drift}")
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()