from pathlib import Path
import json
from .catalog import inspect

if __name__ == "__main__":
    print(json.dumps(inspect(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
