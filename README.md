# GradientFlow

[![CI](https://github.com/kapildhinwa2010-dev/GradientFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/GradientFlow/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/GradientFlow/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/GradientFlow/actions/workflows/codeql.yml)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Screen gradient-flow health of stacked weight layers via condition numbers.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

gradient_health() computes a static spectral proxy per layer: the
condition number bounds how unevenly gradients scale across input
directions, and the module labels risk low (<1e3), medium (<1e6) or
high. flow_report() runs this per layer and names the highest-risk
layer. No actual gradients are tracked - the header says so honestly.

## Install

    pip install -r requirements.txt

## Quick start

    from gradientflow import flow_report

    report = flow_report(layers)
    print(report["highest_risk"], report[report["highest_risk"]]["risk"])

## API

- gradient_health(weights) - condition_number, effective_rank, risk
- flow_report(layers) - {name: health} plus highest_risk
- layer_spectrum, effective_rank, layer_report, analyze_model,
  find_bottlenecks (shared spectral base)

## Honest results and limitations

- This is a screening signal computed from weights alone, not a
  measurement of gradients during training. It cannot see optimizer,
  LR schedule, or data effects.
- The risk thresholds (1e3 / 1e6) are conventions, not calibrated
  results; no experiment here links labels to actual training
  instability.
- condition_number is inf for numerically singular layers; flow_report
  handles it but downstream code should too.

## License

MIT - see [LICENSE](LICENSE).
