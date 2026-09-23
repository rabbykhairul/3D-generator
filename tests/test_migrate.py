from pathlib import Path

from sqlalchemy import create_engine, inspect

from eyewear_vto.config import Settings
from eyewear_vto.migrate import main


def test_migration_creates_job_table(tmp_path: Path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'jobs.sqlite'}"
    settings = Settings(database_url=database_url, _env_file=None)
    monkeypatch.setattr("eyewear_vto.migrate.get_settings", lambda: settings)

    main()

    inspector = inspect(create_engine(database_url))
    assert "generation_jobs" in inspector.get_table_names()
