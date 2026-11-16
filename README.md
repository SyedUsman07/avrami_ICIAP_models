# Estimation of Particle Growth Kinetics via Physics-Informed Neural Networks

Code accompanying the ICIAP 2025 paper *"Estimation of Particle Growth Kinetics via Physics-Informed Neural Networks"*.

The notebooks use Physics-Informed Neural Networks (PINNs) to model crystallization / particle-growth kinetics with the **Avrami (JMAK) equation**, combined with an Arrhenius rate constant:

$$
X(t) = 1 - \exp\left(-(k\,t)^{n}\right), \qquad k = A \exp\left(-\frac{E_a}{R\,T}\right)
$$

The network fits the conversion curve $X(t)$ from a few experimental points and, at the same time, estimates the kinetic parameters: activation energy $E_a$, pre-exponential factor $A$ and Avrami exponent $n$.

## Repository structure

| File | Description |
|---|---|
| `models.py` | PyTorch models: an MLP (`Model`) and an LSTM (`ModelRNN`) with a shared backbone and separate heads for the prediction `d` and the parameters `Ea`, `A`, `n` (Softplus keeps them positive). |
| `sonawane_etal.py` | Experimental conversion-vs-time data at 160, 170, 180 and 190 °C, digitized from Sonawane et al., *Ind. Eng. Chem. Res.* ([10.1021/acs.iecr.3c00322](https://pubs.acs.org/doi/10.1021/acs.iecr.3c00322), Fig. 11b). |
| `1_avrami_base_exp.ipynb` | Tutorial: PINN on the Avrami equation with dense, noise-free data and known `Ea`, `A`. |
| `2_avrami_simple_simulated_explorative.ipynb` | Same setup, but `Ea` and `A` are learned; explores the parameter-scaling problem. |
| `3_avrami_realistic_simulated.ipynb` | Realistic simulated data (few points plus initial conditions), with the scaling issue solved. |
| `4_avrami_real_data_v1*.ipynb` | Real data: `Ea`, `A` (and `n` in `+n`) are output by the network at every time step (time-dependent, with uncertainty). `_unscaled` is the version without parameter scaling. |
| `4_avrami_real_data_v2+n*.ipynb`, `4_avrami_real_data_v3*.ipynb` | Real data: `Ea`, `A`, `n` are global learnable `nn.Parameter`s. `_190deg` variants test at 190 °C. |
| `5_avrami_real_data_v2.ipynb` | `Ea` and `A` are learned only through the physics (PINN) loss. |
| `6_avrami_real_data_paper.ipynb` | Main experiments used for the paper (`_nofit`: variant without the fitting step). |
| `7_avrami_modified_conc-data.ipynb`, `7_avrami_modified_temp-data.ipynb` | Modified Avrami experiments on concentration and temperature data. |
| `8_avrami_modified_conc-data.ipynb` | Modified Avrami on the Sonawane et al. concentration data (`sonawane_etal.py`). |

## Requirements

- Python 3
- PyTorch
- NumPy, SciPy, scikit-learn, Matplotlib
- Jupyter

```bash
pip install torch numpy scipy scikit-learn matplotlib jupyter
```

## Usage

Open the notebooks in Jupyter and run them in order. The numbering follows the development of the method, from the simple tutorial (1) to the paper experiments (6–8).

```bash
jupyter notebook
```

> Note: `8_avrami_modified_conc-data.ipynb` loads a `study.pkl` file, which is not in this repository.

## Citation

If you use this code, please cite:

> U. Syed, E. Radicchi, P. Marone, F. Enrichi, A. Speghini, F. Semenzin, A. Scarsini, F. Setti, F. Cunico.
> **Estimation of Particle Growth Kinetics via Physics-Informed Neural Networks.**
> In *Image Analysis and Processing – ICIAP 2025*, Lecture Notes in Computer Science, pp. 462–474. Springer Nature Switzerland, 2026.
> [https://doi.org/10.1007/978-3-032-10192-1_38](https://link.springer.com/chapter/10.1007/978-3-032-10192-1_38)

```bibtex
@inproceedings{syed2026estimation,
  title     = {Estimation of Particle Growth Kinetics via Physics-Informed Neural Networks},
  author    = {Syed, Usman and Radicchi, Eros and Marone, Paolo and Enrichi, Francesco and Speghini, Adolfo and Semenzin, Filiberto and Scarsini, Alberto and Setti, Francesco and Cunico, Federico},
  booktitle = {Image Analysis and Processing -- ICIAP 2025},
  series    = {Lecture Notes in Computer Science},
  pages     = {462--474},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  doi       = {10.1007/978-3-032-10192-1_38}
}
```
