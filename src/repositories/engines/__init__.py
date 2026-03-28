"""Database engine plugins — registry and factory.

Each engine implements the DatabaseEngine protocol defined in interface.py.
The ENGINES registry maps string names to engine classes, and get_engine()
is the factory function used by BaseRepository to create engine instances.

To add a new engine (e.g., MySQL, PostgreSQL):
1. Create a new file (e.g., mysql.py) implementing all DatabaseEngine methods
2. Import the class here
3. Add it to the ENGINES registry dict below

Usage:
    from src.repositories.engines import get_engine
    engine = get_engine("sqlite", db_path="db/logistics.db")
    engine = get_engine("mysql", host="localhost", user="root", ...)
"""

from src.repositories.engines.interface import DatabaseEngine
from src.repositories.engines.sqlite import SQLiteEngine

# ── Engine Registry ──────────────────────────────────────────────────
# Maps engine name strings to their implementation classes.
# Add new engines here as they are implemented.
ENGINES: dict[str, type] = {
    "sqlite": SQLiteEngine,
}


def get_engine(engine_name: str, **kwargs) -> DatabaseEngine:
    """Factory: create and return an engine instance by name.

    This is the single entry point for engine creation. BaseRepository
    calls this during initialization to get the appropriate engine.

    Args:
        engine_name: One of the keys in ENGINES ("sqlite", "mysql", etc.)
        **kwargs: Engine-specific configuration passed to the constructor.
            For SQLite: db_path (str)
            For MySQL: host, user, password, database, etc.

    Returns:
        A DatabaseEngine implementation instance.

    Raises:
        ValueError: If engine_name is not registered in the ENGINES dict.
    """
    engine_class = ENGINES.get(engine_name)
    if engine_class is None:
        available = ", ".join(ENGINES.keys())
        raise ValueError(
            f"Unknown database engine: '{engine_name}'. Available: {available}"
        )
    return engine_class(**kwargs)


__all__ = ["DatabaseEngine", "get_engine", "ENGINES"]
