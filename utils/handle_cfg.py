
from pathlib import Path
from hydra import initialize_config_dir, compose

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config")
def load_config():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
        cfg = compose(config_name="config")
    return cfg

cfg = load_config()