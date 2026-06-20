# fault-rmt-phase-framework

**From Hypothesis to Reality: Scale-Dependent Universality Classes in Earth's Seismotectonic Rhythms**

A complete random-matrix-theory (RMT) investigation of earthquake timing — from a forward-modeling hypothesis, through empirical validation on real USGS data across eight global fault zones, to a scale-dependent resolution.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

**Author:** Ruqing Chen · GUT Geoservice Inc., Montreal · ruqing@hotmail.com

---

## The Scientific Arc

| Part | Content | Data | Finding |
|------|---------|------|---------|
| **I — Hypothesis** | Forward modeling | Synthetic | Predicts a GOE→Poisson repulsion gradient across tectonic regimes |
| **II — Empirical test** | 8 fault zones | Real USGS/ANSS | Hypothesis **rejected**: declustered mainshocks collapse to Poisson |
| **III — Resolution** | Cluster isolation + Omori fit | Real | Memory lives in **Omori clustering**; the **p-exponent** is the true invariant |

## Three-Scale Universality

| Temporal scale | Universality class | Evidence |
|---|---|---|
| Cascade (hours–days) | **Omori clustering** | CV 1.1–3.8, Gamma γ = 0.16–0.69 |
| Mainshock (years–decades) | **Poisson** | ⟨r⟩ ≈ 0.39, all 8 zones |
| Geological (Myr) | **GOE repulsion** | [companion study](https://zenodo.org/records/20766310) |

The Omori **p-exponent** does not collapse and tracks tectonic setting:
**p = 1.28** (dry, simple Parkfield) → **0.6–0.9** (fluid-rich subduction).
Fluids leave their fingerprint in the aftershock decay rate, not in mainshock recurrence.

## Important: Methodology Note

Part I uses **synthetic catalogs** for hypothesis generation and pipeline
validation — this is standard forward modeling, **not** an empirical claim.
Parts II–III use **real USGS/ANSS ComCat catalogs** (1970–2024, public domain).
The paper is explicit about this distinction throughout.

## Repository Structure

```
fault-rmt-phase-framework/
├── README.md  ·  LICENSE  ·  requirements.txt  ·  CITATION.cff  ·  .zenodo.json
├── paper/
│   ├── paper.tex          # merged LaTeX (Parts I–III)
│   ├── paper.pdf          # compiled (10 pp.)
│   └── figs/              # figures embedded by LaTeX
├── code/
│   ├── 01_synthetic_forward_model.py   # Part I: hypothesis + diagnostics
│   ├── 02_synthetic_figures.py
│   ├── 03_real_batch_pipeline.py       # Part II: 4-stage real-data pipeline
│   ├── 04_full_catalog_clustering.py   # Part III: clustering test
│   ├── 05_naf_segment_probe.py         # Part II: NAF segmentation
│   └── 06_omori_cluster_separation.py  # Part III: Omori p-exponent
├── data/
│   ├── synthetic_fault_params.csv      # Part I priors
│   └── {Parkfield,NAF,Alpine,Nankai,Chile,Cascadia,Charlevoix,Longmenshan}_raw_usgs.csv
├── figures/               # standalone PDF copies
└── results/               # JSON/CSV outputs of every analysis
```

## Pipeline (real data, 4 stages)
1. **Magnitude homogenization** → Mw (Scordilis 2006)
2. **Completeness** Mc (maximum-curvature method)
3. **Dual declustering** (Gardner–Knopoff window + ETAS stochastic, 200 realizations)
4. **RMT spectral statistics** + Omori–Utsu MLE fit on isolated clusters

## Reproduce

```bash
pip install -r requirements.txt
cd code
python 01_synthetic_forward_model.py    # Part I  — hypothesis
python 03_real_batch_pipeline.py        # Part II — empirical test
python 04_full_catalog_clustering.py    # Part III — clustering
python 06_omori_cluster_separation.py   # Part III — Omori invariant
```

## Citation

```bibtex
@misc{chen2026faultrmt,
  author = {Chen, Ruqing},
  title  = {From Hypothesis to Reality: Scale-Dependent Universality Classes
            in Earth's Seismotectonic Rhythms},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/Ruqing1963/fault-rmt-phase-framework}
}
```

## Related Work
Companion geological study (geological-epoch GOE repulsion):
[zenodo.org/records/20766310](https://zenodo.org/records/20766310)

## Version History
- **v2.0.0** — Merged the synthetic forward-modeling framework and the real-data
  empirical study into a single hypothesis→test→resolution paper. Supersedes the
  two earlier standalone drafts.

## License
[MIT](LICENSE). Earthquake data courtesy USGS/ANSS ComCat (public domain).
