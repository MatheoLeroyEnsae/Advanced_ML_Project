import subprocess
import sys

packages = [
    "torch",
    "transformers",
    "accelerate",
    "scikit-learn",
    "datasets",
    "evaluate",
]


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for pkg in packages:
    try:
        install(pkg)
        print(f"[OK] {pkg} installé")
    except Exception as e:
        print(f"[ERROR] {pkg} n'a pas pu être installé : {e}")
