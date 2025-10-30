import glob
import os

base_path = "langchain-docs/langchain.readthedocs.io/en/latest"
print("Current working dir:", os.getcwd())
print("Path exists:", os.path.exists(base_path))

files = glob.glob(f"{base_path}/**/*.html", recursive=True)
print(f"Found {len(files)} HTML files")
