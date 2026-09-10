from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import shutil
from pathlib import Path

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

output_dir = Path("iris_model")
if output_dir.exists():
    shutil.rmtree(output_dir)

mlflow.sklearn.save_model(model, path=str(output_dir))
print("Done! iris_model folder created.")