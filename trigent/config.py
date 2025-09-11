"""Configuration management for Trigent."""
import toml
from pathlib import Path
from typing import Dict, Any, Optional


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration from config.toml file.
    
    Args:
        config_path: Optional path to config file. If None, looks for config.toml
                    in current directory, then project root.
    
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file cannot be found
        ValueError: If config file is invalid TOML
    """
    # Determine config file path
    if config_path:
        config_file = Path(config_path)
    else:
        # Look for config.toml in current directory first
        config_file = Path("config.toml")
        if not config_file.exists():
            # Look in project root (where this module is located)
            project_root = Path(__file__).parent.parent
            config_file = project_root / "config.toml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_file}. "
            f"Copy config.toml.example to config.toml and configure your settings."
        )
    
    try:
        with open(config_file, 'r') as f:
            return toml.load(f)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML in config file {config_file}: {e}")