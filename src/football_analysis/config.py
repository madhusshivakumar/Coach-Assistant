"""Application settings. Reads environment variables prefixed `FA_` and `.env`."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for football_analysis."""

    model_config = SettingsConfigDict(
        env_prefix="FA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("./data"), description="Root of data/ tree")
    log_level: str = Field(default="INFO", description="structlog level")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.duckdb"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
