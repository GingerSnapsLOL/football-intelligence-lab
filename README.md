# Football Intelligence Lab

Data and modelling toolkit for football match and event analysis.

## Quick start

Development:

```bash
make api
make frontend
```

Docker:

```bash
docker compose up --build
```

Bind-mounts local `data/processed/` and `artifacts/` (no training on startup).

UI: http://localhost:5173 · API: http://localhost:8000

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Getting started

```bash
make install   # sync the environment and install pre-commit hooks
make check     # lint, typecheck and test
make data      # download the StatsBomb development dataset (~350 MB)
```

## Make targets

| Target      | Description                                       |
| ----------- | ------------------------------------------------- |
| `install`   | Sync the environment and install git hooks        |
| `format`    | Apply formatting and safe lint fixes              |
| `lint`      | Check formatting and lint rules (no modification) |
| `typecheck` | Run mypy                                          |
| `test`      | Run pytest with coverage                          |
| `check`     | `lint` + `typecheck` + `test`                     |
| `notebook`  | Launch JupyterLab                                 |
| `data`      | Download the StatsBomb development dataset        |
| `data-list` | List every available competition season           |
| `process`   | Rebuild the processed Parquet tables from raw JSON |
| `queries`   | Run the analytical DuckDB queries                 |
| `features`  | Build the canonical shot modelling dataset         |
| `diagnostics` | Describe shot distance and angle                 |
| `comparisons` | Run the two-sample and contingency test demos    |
| `bootstrap` | Bootstrap football quantities by resampling unit   |
| `permutation` | Permutation tests by exchangeability unit        |
| `api`         | Start FastAPI on :8000                            |
| `frontend`    | Start the Vite UI on :5173                        |
| `docker-up`   | `docker compose up --build`                       |
| `docker-down` | `docker compose down`                             |

## Data

Source: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Files are
downloaded on demand into `data/raw/statsbomb/`, mirroring the upstream layout
byte-for-byte. Raw files are never modified in place and are **not** committed.

The default development subset is two men's international tournaments:

| Competition    | Season | Matches | Shots | Goals |
| -------------- | ------ | ------: | ----: | ----: |
| FIFA World Cup | 2018   |      64 | 1,706 |   183 |
| UEFA Euro      | 2020   |      51 | 1,289 |   155 |
| **Total**      |        | **115** | **2,995** | **338** |

That is ~350 MB and 1,199 players, at an 11.3% baseline goal rate — enough for
exploration, hypothesis testing and a first xG model, while staying at a few
percent of the full open-data repository. Because a tournament gives each player
at most ~7 matches, a club season should be added before the repeated-measures
and hierarchical tasks.

```bash
make data                                                   # default subset
make data-list                                              # available seasons
make data DATA_ARGS="--competition-season 11/27"            # add La Liga 2015/16
make data DATA_ARGS="--competition-season 43/3 --limit-matches 5"
```

Downloads are idempotent: per-match files are skipped once present and the index
files are revalidated by ETag, so re-running `make data` transfers nothing when
everything is current. Use `--force` to refetch.

Data is provided by StatsBomb under the StatsBomb Open Data User Agreement;
attribution is required for any published output derived from it.

## Processed tables

`make process` normalises the raw JSON into Parquet under `data/processed/`
(343 MB of JSON becomes 15 MB). Nothing writes back to `data/raw/`.

| Table      | Rows    | Size    | Grain                                                    |
| ---------- | ------: | ------: | -------------------------------------------------------- |
| `matches`  |     115 |   17 KB | One match; includes the coordinate-fidelity flags         |
| `events`   | 420,489 | 14.6 MB | One event, common attributes only (no per-type payloads)  |
| `shots`    |   2,995 |  233 KB | One shot, payload flattened; `is_shootout` marks period 5 |
| `players`  |   1,199 |   49 KB | One player, with squad-listing counts                     |

Query them with DuckDB reading the Parquet directly — no database file is kept,
so the Parquet stays the single source of truth:

```python
from football_intelligence.data import queries

with queries.connect() as connection:
    print(queries.goals_by_player(connection, limit=10))
    print(connection.execute("SELECT * FROM shots WHERE NOT is_shootout LIMIT 5").df())
```

Shootout penalties are excluded from every query by default, and `statsbomb_xg`
is kept as a benchmark rather than a feature — it is the output of StatsBomb's
own model for the quantity this project predicts.

## Shot dataset

`make features` builds `data/processed/shot_dataset.parquet` — one row per shot,
2,918 rows at a 9.87% goal rate, with penalty shootouts excluded.

Geometry is computed against the real goal mouth, posts at (120, 36) and
(120, 44):

- `shot_distance` — yards to the goal centre (120, 40).
- `shot_angle` — radians subtended at the shot by the two posts,
  `atan2(8·|120−x|, (120−x)² + (y−36)(y−44))`, not a proxy.

One coordinate unit is one yard: penalties are recorded 11.9 units from the goal
line and the penalty mark is 12 yards. Attacking direction is already normalised
toward `x = 120` in every period, so coordinates must **not** be flipped by half.

```python
from football_intelligence.features import shots

dataset = shots.read_shot_dataset()
features = dataset[list(shots.FEATURE_COLUMNS)]   # statsbomb_xg excluded by design
target = dataset[shots.TARGET_COLUMN]
```

## Notebooks

| Notebook                        | Contents                                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `01_data_exploration.ipynb`     | Entities, event types, shot outcomes and class balance; pitch coordinate conventions verified against the StatsBomb specification; clustering, missingness and leakage findings for later tasks |

Run `make data` before executing a notebook, then `make notebook`.

## Layout

```
src/football_intelligence/   package source (src layout)
tests/                       test suite
notebooks/                   exploratory notebooks
data/raw|interim|processed/  datasets (git-ignored)
artifacts/                   model and run artifacts (git-ignored)
```
