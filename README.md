# Quality-Controlled Active Learning for Autonomous Microscopy

This repository contains the code for the paper [*Quality-controlled active learning via Gaussian processes for robust structure–property learning in autonomous microscopy*](https://doi.org/10.1038/s41524-026-02248-x), npj Computational Materials (2026).

It implements **ActiveQC** (Active Learning with Quality Control), a gated active learning framework for robust structure-property learning in autonomous microscopy experiments.

## Overview

ActiveQC combines curiosity-driven acquisition with a physics-informed quality control mechanism to filter out low-fidelity samples during training and acquisition. The framework uses:

- **Simple Harmonic Oscillator (SHO) model fits** to evaluate spectral quality
- **Gaussian Process regression** to model spatial distribution of data fidelity
- **Curiosity-driven sampling** that balances exploration and exploitation

The method is evaluated on two bidirectional structure-property tasks:
- **Im2Spec**: Image-to-Spectrum translation (PFM patch → BEPS spectrum)
- **Spec2Im**: Spectrum-to-Image translation (BEPS spectrum → PFM patch)

## Data

The experiments use paired structural and spectroscopic measurements from ferroelectric thin films:
- **PbTiO3** thin films (pre-acquired BEPS dataset with simulated noise) and separate heterogeneous **PbTiO3** thin films with richer domain structures (real-time AFM deployment)

All data are available on Zenodo: [https://doi.org/10.5281/zenodo.18881435](https://doi.org/10.5281/zenodo.18881435)

- Pre-acquired dataset (`BEPS_1d7um_0009.h5`) → `inputs/data/`
- Real-time AFM deployment runs (`v2`, `v2b`, `v2c`, `v2d`, `v2e`, `v2f`) → `real_time_afm/inputs/data_PTO/`

## Repository Structure

```
├── 1_noise_induction.ipynb          # Simulate spatially localized noise
├── 2_sho_fitting_clean_spectra.py   # SHO fitting on clean spectra
├── 3_sho_fitting_noisy_spectra.py   # SHO fitting on noisy spectra
├── 4_phase_correction.ipynb         # Phase correction for BEPS data
├── 5_exp_im2spec_soloRun.ipynb      # Im2Spec single run experiment
├── 6_exp_im2spec_multiRun.ipynb     # Im2Spec multi-trial experiments
├── 7_exp_im2spec_multiRun.py        # Im2Spec multi-trial (script version)
├── 8_exp_im2spec_results.ipynb      # Im2Spec results analysis
├── 9_exp_spec2im_soloRun.ipynb      # Spec2Im single run experiment
├── 10_exp_spec2im_multiRun.ipynb    # Spec2Im multi-trial experiments
├── 11_exp_spec2im_multiRun.py       # Spec2Im multi-trial (script version)
├── 12_exp_spec2im_results.ipynb     # Spec2Im results analysis
├── 13_exp_im2spec_sensitivity.py    # sensitivity analysis with Im2Spec
├── 14_exp_spec2im_sensitivity.py    # sensitivity analysis with Spec2Im
├── models.py                        # Neural network architectures
├── utils.py                         # Utility functions
├── inputs/                          # Input data directory
└── real_time_afm/                   # Real-time AFM deployment code
    ├── execution_im2spec_PTO.ipynb  # ActiveQC run on the microscope
    ├── analysis_im2spec_PTO.ipynb   # Analysis of the real-time runs
    ├── models.py
    ├── utils.py
    └── inputs/data_PTO/             # Real-time run data
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- Optuna
- Matplotlib
- [AtomAI](https://github.com/pycroscopy/atomai)
- [Im2Spec](https://github.com/ziatdinovmax/im2spec)

## Usage

### Data Preparation
1. Run `1_noise_induction.ipynb` to simulate spatially localized noise
2. Run `2_sho_fitting_clean_spectra.py` and `3_sho_fitting_noisy_spectra.py` for SHO fitting
3. Run `4_phase_correction.ipynb` for phase correction

### Running Experiments
For Im2Spec experiments:
```bash
python 7_exp_im2spec_multiRun.py
```

For Spec2Im experiments:
```bash
python 11_exp_spec2im_multiRun.py
```

For sensitivity analysis of the quality threshold:
```bash
python 13_exp_im2spec_sensitivity.py
python 14_exp_spec2im_sensitivity.py
```

### Results Analysis
Use `8_exp_im2spec_results.ipynb` and `12_exp_spec2im_results.ipynb` to analyze results.

### Real-time AFM Deployment
`real_time_afm/execution_im2spec_PTO.ipynb` runs ActiveQC on the microscope (requires the instrument and [AEcroscoPy](https://github.com/pycroscopy/aecroscopy)). `real_time_afm/analysis_im2spec_PTO.ipynb` analyzes the recorded runs in `real_time_afm/inputs/data_PTO/`.

## Acquisition Strategies Compared

| Strategy | Description |
|----------|-------------|
| **Random** | Uniform random sampling (baseline) |
| **Active** | Curiosity-driven acquisition without quality control |
| **ActiveMT** | Multi-task learning with auxiliary reconstruction |
| **ActiveQC** | Quality-controlled gated sampling (proposed) |

## Citation

If you use this code, please cite:

```bibtex
@article{chowdhury2026activeqc,
  title={Quality-controlled active learning via Gaussian processes for robust structure--property learning in autonomous microscopy},
  author={Chowdhury, Jawad and Narasimha, Ganesh and Yang, Jan-Chi and Funakubo, Hiroshi and Ehara, Yoshitaka and Liu, Yongtao and Vasudevan, Rama},
  journal={npj Computational Materials},
  year={2026},
  doi={10.1038/s41524-026-02248-x}
}
```

## License

See [LICENSE](LICENSE) for details.
