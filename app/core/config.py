from dotenv import load_dotenv
import os
from app.core.logger import get_logger, setup_logging
from app.schemas.config_schema import Settings

load_dotenv()

# 1. Create the configuration instance, which will read from the environment
config = Settings()

# 2. Set up logging based on the loaded configuration
setup_logging(config.LOGLEVEL)

# 3. Get a logger for this file, now that logging is configured
logger = get_logger(__name__)


def ensure_data_dir(path: str) -> str:
    """Ensure the parent directory of the given path exists and return the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return path
