from __future__ import annotations

from sqlalchemy import create_engine

from eyewear_vto.config import get_settings
from eyewear_vto.sql_jobs import metadata


def main() -> None:
    settings = get_settings()
    if not settings.database_url:
        return
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    metadata.create_all(engine)
    engine.dispose()


if __name__ == "__main__":
    main()
