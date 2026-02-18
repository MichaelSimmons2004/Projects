import yaml
import os
from pathlib import Path
from cryptography.fernet import Fernet

BASE_DIR = Path(__file__).parent
KEY_FILE = BASE_DIR / 'secret.key'
CONFIG_FILE = BASE_DIR / 'config.enc'

DEFAULT_CONFIG = {
    'response_style': 'detailed',
    'ethical_guidelines': 'none',
    'model': 'placeholder',
    'plugins_enabled': True,
}

def _load_or_create_cipher() -> Fernet:
    """Load the encryption key from disk, creating it if absent."""
    if not KEY_FILE.exists():
        key = Fernet.generate_key()
        KEY_FILE.write_bytes(key)
    return Fernet(KEY_FILE.read_bytes())

cipher = _load_or_create_cipher()

def save_config(config_dict: dict) -> None:
    yaml_str = yaml.dump(config_dict)
    encrypted = cipher.encrypt(yaml_str.encode())
    CONFIG_FILE.write_bytes(encrypted)

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        encrypted = CONFIG_FILE.read_bytes()
        yaml_str = cipher.decrypt(encrypted).decode()
        return yaml.safe_load(yaml_str)
    except Exception:
        # Key mismatch or corrupted file — reset to defaults.
        CONFIG_FILE.unlink(missing_ok=True)
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)