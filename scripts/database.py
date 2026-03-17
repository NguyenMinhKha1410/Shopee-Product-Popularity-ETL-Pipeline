from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from scripts.config import Settings, get_settings


def get_engine(settings: Settings | None = None) -> Engine:
    resolved_settings = settings or get_settings()
    return create_engine(
        resolved_settings.sqlalchemy_url,
        future=True,
        pool_pre_ping=True,
    )
