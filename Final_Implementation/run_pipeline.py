import os
import subprocess
import sys
import time
from pathlib import Path

import os
# Define project directory as the directory containing this script
PROJECT_DIR = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_DIR / "src"
K8S_FILE = PROJECT_DIR / "k8s" / "fastapi-deployment.yaml"

def get_pip_path(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    else:
        return venv_dir / "bin" / "pip"

def get_python_path(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"

# Step 1: Set up virtual environment and install dependencies
def setup_venv():
    print("🔧 Setting up virtual environment...")
    venv_dir = PROJECT_DIR / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)])

    pip = get_pip_path(venv_dir)
    subprocess.run([str(pip), "install", "--upgrade", "pip"])
    subprocess.run([str(pip), "install", "-r", str(PROJECT_DIR / "requirements.txt")])
    print("✅ Virtual environment and dependencies installed.")
    return venv_dir

# Step 2: Start MLflow tracking server
def start_mlflow_tracking():
    print("🚀 Starting MLflow tracking server...")
    os.makedirs(PROJECT_DIR / "mlruns", exist_ok=True)
    # Use mlflow executable from the virtual environment
    mlflow_exe = str(PROJECT_DIR / "venv" / "Scripts" / "mlflow.exe")
    subprocess.Popen([
        mlflow_exe, "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", str(PROJECT_DIR / "mlruns"),
        "--host", "127.0.0.1",
        "--port", "5000"
    ])
    print("📡 MLflow tracking server running at http://127.0.0.1:5000")
    time.sleep(5)  # Allow server to start

# Step 3: Run MLflow training job
def run_training():
    print("📈 Training the model via MLflow...")
    subprocess.run(["mlflow", "run", ".", "-P", "alpha=0.5"], cwd=PROJECT_DIR)

# Step 4: Serve model with FastAPI
def serve_model():
    print("🌐 Starting FastAPI server...")
    subprocess.Popen(["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"], cwd=PROJECT_DIR)
    print("✅ FastAPI running at http://127.0.0.1:8000")
    time.sleep(3)

# Step 5: Run unit tests
def run_tests():
    print("🧪 Running tests...")
    subprocess.run(["pytest", "tests"], cwd=PROJECT_DIR)

# Step 6: Deploy to Kubernetes
def deploy_to_k8s():
    print("📦 Deploying to Kubernetes...")
    placeholder = "/path/to/ml-project/src"
    real_path = str(SRC_DIR)

    # Read and update path in YAML
    with open(K8S_FILE, "r") as f:
        content = f.read().replace(placeholder, real_path)

    with open(K8S_FILE, "w") as f:
        f.write(content)

    subprocess.run(["kubectl", "apply", "-f", str(K8S_FILE)])
    print("✅ Kubernetes deployment applied.")
    print("ℹ️ Access via `minikube service fastapi-service` or your cluster's external IP.")

def main():
    setup_venv()
    start_mlflow_tracking()
    run_training()
    serve_model()
    run_tests()

    if input("Deploy to Kubernetes? (y/n): ").lower() == "y":
        deploy_to_k8s()

    print("\n🎉 Pipeline completed!")
    print("🔍 Test prediction: POST to http://127.0.0.1:8000/predict with JSON {\"x\": 5}")

if __name__ == "__main__":
    main()
