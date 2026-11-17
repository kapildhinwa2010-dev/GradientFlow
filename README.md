# GradientFlow

GradientFlow is a small Python module for inspecting the singular-value properties of named weight matrices. It reports per-matrix norms, a condition number and a singular-value-based effective-rank estimate, and provides a heuristic label intended as a static gradient-flow screening signal.

This is an early, narrowly scoped utility (version `0.1.0`), not a neural-network training or monitoring framework. It analyzes matrix values supplied by the caller; it does not run a model, observe training, or calculate actual gradients.

## Requirements

- Python 3.9 or later
- NumPy 1.24 or later

Install the project from a local checkout with:

```bash
python -m pip install .
```

To install the test dependency as well, use the repository's requirements file:

```bash
python -m pip install -r requirements.txt
```

There is no command-line interface. Use the module from Python and provide non-empty mappings of layer names to two-dimensional NumPy-compatible matrices.

## Example

```python
import numpy as np
from gradientflow import flow_report

layers = {
    "projection": np.array([[1.0, 0.0], [0.0, 0.5]]),
    "output": np.eye(2),
}

report = flow_report(layers)
print(report["highest_risk"])
print(report["projection"])
```

`flow_report` returns one health dictionary per input name, plus `highest_risk`, the name of the layer with the largest condition number. A matrix with an effectively zero smallest singular value has an infinite condition number; downstream code should account for that value.

## Python API

- `layer_spectrum(weights)` converts the input to `float64` and returns its singular values in descending order. The input must be two-dimensional.
- `effective_rank(weights, energy=0.99)` returns the smallest number of leading singular values whose cumulative **sum of singular values** reaches the requested fraction. Despite the parameter name, the implementation does not sum squared singular values. `energy` must be in `(0, 1]`; a zero matrix returns rank `1`.
- `layer_report(weights)` returns `shape`, `frobenius_norm`, `spectral_norm`, `condition_number`, and `effective_rank`. It rejects empty or non-2-D matrices. The condition number is reported as infinity when the smallest singular value is at or below the implementation's machine-precision threshold.
- `analyze_model(layers)` applies `layer_report` to a non-empty `{name: matrix}` mapping. It does not load or inspect a model object.
- `find_bottlenecks(reports, top_k=1)` returns names ordered by ascending effective rank, with names used to break ties. An empty report mapping returns an empty list.
- `gradient_health(weights)` returns `condition_number`, `effective_rank`, and a heuristic `risk` label: `low` below `1e3`, `medium` below `1e6`, and `high` otherwise.
- `flow_report(layers)` applies `gradient_health` to each matrix and adds the `highest_risk` name.

The risk cutoffs are fixed heuristics in the source; this repository contains no calibration study showing that the labels predict training behavior. Treat them as screening labels, not diagnoses or measurements of gradient flow.

## Tests and automation

The tests were run locally against commit `47575c96821c1b6a3966ef1b8e6852df2152ce37` with:

```bash
python -m pytest -q -p no:cacheprovider
```

Result: **16 passed**. The GitHub Actions CI workflow is configured to install `requirements.txt` and run pytest on Python 3.10, 3.11, and 3.12; this local result does not establish the status of remote workflow runs.

To run the tests after installing the repository requirements:

```bash
python -m pytest -q
```

## Limitations

- Inputs are individual two-dimensional matrices or mappings of names to matrices; there is no adapter for model frameworks, checkpoints, or live training loops.
- The analysis uses weights alone. It does not inspect gradients, activations, optimizers, data, or learning-rate schedules.
- Condition number and effective rank are matrix-level numerical summaries; the `risk` thresholds are not validated against training outcomes here.
- This repository provides no CLI, training code, benchmark suite, or measured performance claims.

## License

MIT License. See [LICENSE](LICENSE) for the complete terms and copyright notice.
