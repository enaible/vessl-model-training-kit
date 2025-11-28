from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


# Get the project root directory (where .env file is located)
_project_root = Path(__file__).parent.absolute()
_env_file = _project_root / ".env"


class Settings(BaseSettings):
    # API Keys (optional)
    openai_api_key: Optional[str] = 'sk-proj-1234567890'
    gemini_api_key: Optional[str] = None

    # VESSL settings (required for process_answers to work)
    PROCESS_ANSWER: str
    VESSL_URL: str

    model_config = SettingsConfigDict(
        env_file=str(_env_file),  # Use absolute path to .env file
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow case-insensitive environment variable matching
        extra="ignore"  # Ignore extra fields in .env that aren't defined in the model
    )


def load_settings():
    settings = Settings()
    return settings


if __name__ == "__main__":
    settings = load_settings()
