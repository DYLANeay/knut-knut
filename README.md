# knut-knut — IKT110 Lab 2

Route-planning lab for Knut-Knut Transport: a Jupyter notebook (`handin1.ipynb`)
and a small Flask web app (`knut_knut_app.py`).

## Dependencies

All dependencies live in `pyproject.toml` — do **not** run `pip install` inside
the notebook.

| Package | Used for |
| --- | --- |
| `numpy` | arrays / numeric work |
| `plotly`, `pandas` | line & scatter plots (Plotly Express needs pandas) |
| `tqdm` | progress bars |
| `flask` | the web app |
| `jupyterlab`, `ipywidgets` | running the notebook, rendering Plotly figures |

Python 3.14 (pinned in `.python-version`).

## Install

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
```

That creates `.venv/` with the exact versions from `uv.lock`.

## Run

Notebook:

```bash
uv run jupyter lab handin1.ipynb
```

Web app (then open http://127.0.0.1:5000):

```bash
uv run python knut_knut_app.py
```

## Adding a dependency

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock` — commit both.
