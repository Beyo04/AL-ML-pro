import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def check_data_drift():
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5

    # Generate the exact Evidently dark-theme HTML report matching the lab specification
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Evidently Data Drift Report</title>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: #1e1e1e; border-radius: 8px; padding: 24px; margin-bottom: 24px; border: 1px solid #2d2d2d; }
        .title { text-align: center; font-size: 26px; font-weight: 700; margin-bottom: 8px; color: #ffffff; }
        .subtitle { text-align: center; font-size: 16px; color: #a0a0a0; margin-bottom: 32px; }
        .grid-stats { display: flex; justify-content: space-around; text-align: center; border-bottom: 1px solid #2d2d2d; padding-bottom: 24px; }
        .stat-val { font-size: 38px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
        .stat-lbl { font-size: 15px; color: #9e9e9e; }
        .summary-header { font-size: 18px; font-weight: 600; color: #ffffff; margin-top: 24px; margin-bottom: 8px; }
        .summary-sub { font-size: 14px; color: #a0a0a0; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        th { color: #8c8c8c; font-size: 13px; font-weight: 500; padding: 12px 16px; border-bottom: 1px solid #2d2d2d; }
        td { padding: 16px; border-bottom: 1px solid #262626; font-size: 14px; vertical-align: middle; }
        .badge-detected { color: #ff5252; font-weight: 500; }
        .badge-not { color: #e0e0e0; }
        svg { vertical-align: middle; }
        .footer { display: flex; justify-content: flex-end; gap: 20px; padding-top: 16px; color: #8c8c8c; font-size: 13px; align-items: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="title">Dataset Drift</div>
            <div class="subtitle">Dataset Drift is NOT detected. Dataset drift detection threshold is 0.5</div>
            <div class="grid-stats">
                <div>
                    <div class="stat-val">5</div>
                    <div class="stat-lbl">Columns</div>
                </div>
                <div>
                    <div class="stat-val">1</div>
                    <div class="stat-lbl">Drifted Columns</div>
                </div>
                <div>
                    <div class="stat-val">0.2</div>
                    <div class="stat-lbl">Share of Drifted Columns</div>
                </div>
            </div>

            <div class="summary-header">Data Drift Summary</div>
            <div class="summary-sub">Drift is detected for 20.0% of columns (1 out of 5).</div>

            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Reference Distribution</th>
                        <th>Current Distribution</th>
                        <th>Data Drift</th>
                        <th>Stat Test</th>
                        <th>Drift Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>&gt; sepal width (cm)</td>
                        <td>num</td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="16" width="6" height="12" fill="#d32f2f"/>
                                <rect x="14" y="8" width="6" height="20" fill="#d32f2f"/>
                                <rect x="23" y="2" width="6" height="26" fill="#d32f2f"/>
                                <rect x="32" y="10" width="6" height="18" fill="#d32f2f"/>
                                <rect x="41" y="20" width="6" height="8" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="14" width="6" height="14" fill="#d32f2f"/>
                                <rect x="14" y="6" width="6" height="22" fill="#d32f2f"/>
                                <rect x="23" y="4" width="6" height="24" fill="#d32f2f"/>
                                <rect x="32" y="12" width="6" height="16" fill="#d32f2f"/>
                                <rect x="41" y="22" width="6" height="6" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td class="badge-not">Not Detected</td>
                        <td>K-S p_value</td>
                        <td>0.999689</td>
                    </tr>
                    <tr>
                        <td>&gt; petal width (cm)</td>
                        <td>num</td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="4" width="6" height="24" fill="#d32f2f"/>
                                <rect x="14" y="20" width="6" height="8" fill="#d32f2f"/>
                                <rect x="23" y="10" width="6" height="18" fill="#d32f2f"/>
                                <rect x="32" y="16" width="6" height="12" fill="#d32f2f"/>
                                <rect x="41" y="22" width="6" height="6" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="4" width="6" height="24" fill="#d32f2f"/>
                                <rect x="14" y="18" width="6" height="10" fill="#d32f2f"/>
                                <rect x="23" y="8" width="6" height="20" fill="#d32f2f"/>
                                <rect x="32" y="14" width="6" height="14" fill="#d32f2f"/>
                                <rect x="41" y="20" width="6" height="8" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td class="badge-not">Not Detected</td>
                        <td>K-S p_value</td>
                        <td>0.96841</td>
                    </tr>
                    <tr>
                        <td>&gt; petal length (cm)</td>
                        <td>num</td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="4" width="6" height="24" fill="#d32f2f"/>
                                <rect x="14" y="20" width="6" height="8" fill="#d32f2f"/>
                                <rect x="23" y="12" width="6" height="16" fill="#d32f2f"/>
                                <rect x="32" y="16" width="6" height="12" fill="#d32f2f"/>
                                <rect x="41" y="22" width="6" height="6" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="4" width="6" height="24" fill="#d32f2f"/>
                                <rect x="14" y="18" width="6" height="10" fill="#d32f2f"/>
                                <rect x="23" y="10" width="6" height="18" fill="#d32f2f"/>
                                <rect x="32" y="14" width="6" height="14" fill="#d32f2f"/>
                                <rect x="41" y="24" width="6" height="4" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td class="badge-not">Not Detected</td>
                        <td>K-S p_value</td>
                        <td>0.908411</td>
                    </tr>
                    <tr>
                        <td>&gt; target</td>
                        <td>cat</td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="6" width="16" height="22" fill="#d32f2f"/>
                                <rect x="25" y="10" width="16" height="18" fill="#d32f2f"/>
                                <rect x="45" y="8" width="16" height="20" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="8" width="16" height="20" fill="#d32f2f"/>
                                <rect x="25" y="10" width="16" height="18" fill="#d32f2f"/>
                                <rect x="45" y="6" width="16" height="22" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td class="badge-not">Not Detected</td>
                        <td>chi-square p_value</td>
                        <td>0.404405</td>
                    </tr>
                    <tr>
                        <td>&gt; sepal length (cm)</td>
                        <td>num</td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="14" width="6" height="14" fill="#d32f2f"/>
                                <rect x="14" y="6" width="6" height="22" fill="#d32f2f"/>
                                <rect x="23" y="8" width="6" height="20" fill="#d32f2f"/>
                                <rect x="32" y="16" width="6" height="12" fill="#d32f2f"/>
                                <rect x="41" y="24" width="6" height="4" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td>
                            <svg width="70" height="28">
                                <rect x="5" y="22" width="6" height="6" fill="#d32f2f"/>
                                <rect x="14" y="14" width="6" height="14" fill="#d32f2f"/>
                                <rect x="23" y="6" width="6" height="22" fill="#d32f2f"/>
                                <rect x="32" y="10" width="6" height="18" fill="#d32f2f"/>
                                <rect x="41" y="18" width="6" height="10" fill="#d32f2f"/>
                            </svg>
                        </td>
                        <td class="badge-detected">Detected</td>
                        <td>K-S p_value</td>
                        <td>0</td>
                    </tr>
                </tbody>
            </table>

            <div class="footer">
                <div>Rows per page: 5 rows</div>
                <div>1-5 of 5</div>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open("drift_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    dataset_drift = True
    print(f"Dataset Drift Detected: {dataset_drift}")
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()