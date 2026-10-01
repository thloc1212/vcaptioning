"""YouCook2 PELT boundary quality, using released UniVL features."""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""# YouCook2 boundary experiment

    In the molab terminal, run `python -m pip install -r requirements-segmentation.txt`
    and `python scripts/download_yc2_univl.py` from the repository root.
    The download streams the official 4 GB archive and keeps only UniVL features.
    """)
    return (mo,)


@app.cell
def _(mo):
    penalty = mo.ui.slider(start=1, stop=100, step=1, value=10, label="PELT penalty")
    limit = mo.ui.slider(start=10, stop=500, step=10, value=50, label="Maximum videos")
    run = mo.ui.run_button(label="Evaluate")
    mo.vstack([penalty, limit, run])
    return penalty, limit, run


@app.cell
def _(mo, penalty, limit, run):
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    features = root / "data/features/yc2/UniVL_features/UniVL_visual"
    mo.stop(not run.value, mo.md("Press **Evaluate** after downloading features."))
    mo.stop(not features.is_dir(), mo.md("UniVL features are missing. Run the download command above."))
    from temporal_segmentation.evaluate_pelt import evaluate

    result = evaluate(root / "data/yc2/captiondata/yc2_val.json", features,
                      penalty=penalty.value, limit=limit.value)
    mo.md("```json\n" + json.dumps(result, indent=2) + "\n```")
    return


if __name__ == "__main__":
    app.run()
