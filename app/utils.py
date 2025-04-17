import os
import logging

logger = logging.getLogger(__name__)

def get_environment_variables():
    """Get all environment variables required by the application."""
    env_vars = {
        "MODEL_PATH": os.environ.get("MODEL_PATH", "models/model.joblib"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    }
    
    logger.info(f"Environment variables: {env_vars}")
    return env_vars

def setup_logging():
    """Configure logging for the application."""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Logging configured with level: {log_level}")