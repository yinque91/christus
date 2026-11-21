"""Version constants.

SCHEMA_VERSION is the on-disk SQLite/checkpoint schema version. Bumping it
requires a migration entry in wphr.memory.schema.MIGRATIONS.
"""

__version__ = "1.0.0"
SPEC_VERSION = "MASTER SPEC 1.0"
#: on-disk database / checkpoint schema version
SCHEMA_VERSION = 2
#: configuration-file format version (independent from the database schema)
CONFIG_VERSION = 1
