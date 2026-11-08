-- Database creation only. Alembic owns every table, index and constraint in
-- `beacon` (see alembic/versions/) -- putting DDL here would give the schema two
-- sources of truth that drift the moment a migration lands.
--
-- Runs once, from /docker-entrypoint-initdb.d, against the `postgres` database
-- on a fresh data volume. POSTGRES_DB is deliberately pinned to `postgres` so
-- the entrypoint does not create `beacon` itself and this file stays the single
-- place that names the application databases.

-- The application database. Owned by the role the entrypoint already created
-- from POSTGRES_USER, so `alembic upgrade head` connects with no further grants.
CREATE DATABASE beacon OWNER beacon;

-- Phoenix stores its traces in Postgres rather than a container-local SQLite
-- file, so its own logical database keeps those tables clear of Alembic's.
CREATE DATABASE phoenix OWNER beacon;
