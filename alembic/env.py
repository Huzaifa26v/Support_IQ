# Import Python logging configuration.
from logging.config import fileConfig

# Import SQLAlchemy tools.
from sqlalchemy import engine_from_config
from sqlalchemy import pool

# Import Alembic context.
from alembic import context

# Import SupportIQ application settings.
from app.core.config import settings

# Import the shared SQLAlchemy Base class.
from app.db.base import Base

# Import models so SQLAlchemy registers their tables.
# Without these imports, Alembic may not detect the models.
from app.models import User, Ticket  # noqa: F401


# ==========================================================
# 1. ALEMBIC CONFIGURATION
# ==========================================================

# Get the Alembic configuration object.
config = context.config

# Configure logging using alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the database URL defined in SupportIQ settings.
# Replace percent signs to prevent ConfigParser interpolation issues.
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("%", "%%"),
)

# Connect Alembic with SQLAlchemy model metadata.
# This allows Alembic to detect changes in our database models.
target_metadata = Base.metadata


# ==========================================================
# 2. OFFLINE MIGRATIONS
# ==========================================================

def run_migrations_offline() -> None:
    """
    Run migrations without creating a database connection.

    Alembic generates SQL statements using the configured URL.
    """

    # Get the database URL.
    url = config.get_main_option("sqlalchemy.url")

    # Configure Alembic for offline migration generation.
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    # Start and execute the migration transaction.
    with context.begin_transaction():
        context.run_migrations()


# ==========================================================
# 3. ONLINE MIGRATIONS
# ==========================================================

def run_migrations_online() -> None:
    """
    Run migrations using an actual database connection.
    """

    # Create a database engine using Alembic configuration.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Connect to the database.
    with connectable.connect() as connection:

        # Configure Alembic with the database connection
        # and SQLAlchemy model metadata.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        # Start and execute the migration transaction.
        with context.begin_transaction():
            context.run_migrations()


# ==========================================================
# 4. SELECT MIGRATION MODE
# ==========================================================

# Run offline migrations when no database connection is required.
if context.is_offline_mode():
    run_migrations_offline()

# Otherwise, run migrations using a live database connection.
else:
    run_migrations_online()