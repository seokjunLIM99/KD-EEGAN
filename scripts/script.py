import subprocess
import sys
from pathlib import Path


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(project_root / 'script.py'), *sys.argv[1:]],
        cwd=str(project_root),
        check=True)
