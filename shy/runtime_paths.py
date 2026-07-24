from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
TRAINING_LOGS_DIR = PROJECT_ROOT / "training_logs"


def dataset_dir(dataset_name: str) -> Path:
    return DATA_DIR / dataset_name


def ensure_runtime_dirs() -> None:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def create_run_dir(base_dir: Path, run_name: str) -> Path:
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
