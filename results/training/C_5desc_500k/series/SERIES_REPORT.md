# 5-descriptor run: physics per 100k checkpoint

Generated from 5 cached checkpoint(s): gen_110k, gen_200k, gen_300k, gen_400k, gen_500k

Everything below comes from the production `gpumd.exe` with the checkpoint's own `nep.txt`. Nothing is fitted except the exchange constants themselves (which are a definition) and parabolic interpolations that locate minima.

## 1. 训练精度与静态磁性

| checkpoint | gen | E MAE (meV/at) | F MAE (eV/Å) | M MAE (μB) | a0 (Å) | B0 (GPa) | m_eq (μB) | Tc (K) | FM lowest |
|---|---|---|---|---|---|---|---|---|---|
| gen_110k | 110000 | 6.53 | 0.087 | 0.021 | 2.8328 | 286.6 | 2.455 | 885 | yes |
| gen_200k | 200000 | 6.34 | 0.086 | 0.021 | 2.8336 | 278.3 | 2.455 | 932 | yes |
| gen_300k | 300000 | 6.36 | 0.085 | 0.021 | 2.8335 | 278.4 | 2.445 | 912 | yes |
| gen_400k | 400000 | 6.35 | 0.085 | 0.021 | 2.8337 | 274.8 | 2.439 | 924 | yes |
| gen_500k | 500000 | 6.36 | 0.084 | 0.021 | 2.8346 | 273.4 | 2.442 | 912 | yes |

Reference values used for the error columns: a0 = 2.866 Å, m = 2.2 μB, Tc = 1043 K, B0 ≈ 170-180 GPa (the model is deliberately stiffer than experiment, see the main report).

## 2. 交换常数 J1..J6（冻结螺旋拟合）

Convention `H = -1/2 Σ_ij J_ij e_i·e_j` with `e_i` a unit vector, so `E(q)/N - E_FM/N = 1/2 Σ_n J_n [z_n - S_n(q)]`; J in meV at the model's own moment, `J/m²` is the same number divided by m².

| checkpoint | J1 (meV) | J2 (meV) | J3 (meV) | J4 (meV) | J5 (meV) | J6 (meV) | fit rms (meV/at) | D_direct (meV·Å²) | D_LSWT (meV·Å²) | LSWT/direct |
|---|---|---|---|---|---|---|---|---|---|---|
| gen_110k | 60.5 | -1.4 | -5.8 | 2.6 | -3.3 | -1.1 | 0.82 | 467 | 456 | 0.993 |
| gen_200k | 55.0 | 4.7 | -2.7 | 0.4 | -1.8 | -0.1 | 0.72 | 476 | 446 | 0.982 |
| gen_300k | 60.1 | -0.3 | -5.4 | 2.2 | -3.0 | -0.9 | 0.74 | 458 | 460 | 1.002 |
| gen_400k | 57.0 | 3.2 | -3.5 | 0.7 | -2.2 | -0.1 | 0.44 | 445 | 438 | 0.997 |
| gen_500k | 55.5 | 4.4 | -2.9 | 0.4 | -1.9 | -0.0 | 0.41 | 441 | 436 | 0.998 |

Shells: n=1: r=2.455 Å, z=8, n=2: r=2.835 Å, z=6, n=3: r=4.009 Å, z=12, n=4: r=4.701 Å, z=24, n=5: r=4.910 Å, z=8, n=6: r=5.669 Å, z=6 (cutoff 6.0 Å).

**D is evaluated three ways** and they must agree: the direct frozen spirals (`hbar ω = 2S[E(q)-E(FM)]/N`, S = m/2) along [001], [110], [111]; the closed form of the LSWT curve `D = (S/6) Σ_n J_n z_n r_n²`; and the naive `ω`-vs-`q²` slope (kept only as a warning, it is biased low by the quartic term). `LSWT/direct` close to 1 is the self-check that the factor S has been applied — the first version of stage C dropped it and was low by 1/S.

## 3. D 偏硬的来源：把磁矩归一化

`D ∝ |S|^p` with p ≈ 2.35 here, so a moment that is ~11% too large inflates D by ~30% on its own. The table separates the two errors.

| checkpoint | m_eq (μB) | D at m_eq | D at 2.2 μB | dlnD/dlnm | excess at m_eq | excess at 2.2 μB |
|---|---|---|---|---|---|---|
| gen_110k | 2.455 | 467 | 353 | 2.35 | 56% | 18% |
| gen_200k | 2.455 | 476 | 362 | 2.35 | 59% | 21% |
| gen_300k | 2.445 | 458 | 361 | 2.19 | 53% | 20% |
| gen_400k | 2.439 | 445 | 359 | 2.08 | 48% | 20% |
| gen_500k | 2.442 | 441 | 359 | 2.02 | 47% | 20% |

The 300 meV·Å² reference is the centre of the 280-320 meV·Å² band that the main report uses for α-Fe. Even renormalised to the experimental moment the model stays above it, so the exchange is genuinely too stiff; the moment error is only part of the story.

## 4. 磁子谱

`ω(q)` along Γ-H-N-Γ-P-H from the LSWT form of the fitted `J_n`, and the direct spiral points along [001]/[110]/[111]. The LSWT curve is stored at 205 path points per checkpoint; the figure `series_magnon_path.png` overlays them.

| checkpoint | ω at H (meV) | ω at N (meV) | ωmax (meV) | D [001] | D [110] | D [111] |
|---|---|---|---|---|---|---|
| gen_110k | 1317 | 542 | 1339 | 467 | 477 | 467 |
| gen_200k | 1352 | 544 | 1104 | 476 | 454 | 455 |
| gen_300k | 1348 | 544 | 1306 | 458 | 487 | 471 |
| gen_400k | 1345 | 540 | 1154 | 445 | 453 | 449 |
| gen_500k | 1337 | 539 | 1105 | 441 | 448 | 446 |

## 5. 各磁态声子谱（每个态在自己的弛豫晶格常数上）

The force constants are periodic with the *magnetic* cell, so each state is displaced in its own cell: FM/DLM primitive 7×7×7 (343 atoms), NC-90 / AFM-G 5×5×5 conventional (250), AFM-A 5×5×5 of the 4-atom cell (500), DLM averaged over 2 random configurations, δ = 0.02 Å central differences.

| checkpoint | state | a0 (Å) | ωmax (THz) | softest (THz) | modes < -1e-3 | modes < -0.1 | ASR before (eV/Å²) |
|---|---|---|---|---|---|---|---|
| gen_110k | FM | 2.8359 | 10.995 | -0.000 | 0 | 0 | 0.23 |
| gen_110k | NC-90 | 2.8415 | 10.357 | -0.003 | 8 | 0 | 11.49 |
| gen_110k | AFM-A | 2.8542 | 10.361 | -0.003 | 12 | 0 | 14.17 |
| gen_110k | AFM-G | 2.8808 | 6.719 | -7.018 | 334 | 324 | 5.13 |
| gen_110k | DLM | 2.8598 | 9.682 | -0.000 | 0 | 0 | 0.13 |
| gen_200k | FM | 2.8369 | 11.070 | -0.000 | 0 | 0 | 0.16 |
| gen_200k | NC-90 | 2.8402 | 10.418 | -0.002 | 8 | 0 | 11.44 |
| gen_200k | AFM-A | 2.8524 | 10.499 | -0.272 | 24 | 10 | 14.70 |
| gen_200k | AFM-G | 2.8944 | 6.781 | -7.479 | 363 | 355 | 5.25 |
| gen_200k | DLM | 2.8582 | 9.768 | 0.000 | 0 | 0 | 0.10 |
| gen_300k | FM | 2.8363 | 10.999 | 0.000 | 0 | 0 | 0.17 |
| gen_300k | NC-90 | 2.8388 | 10.342 | -0.003 | 8 | 0 | 10.78 |
| gen_300k | AFM-A | 2.8513 | 10.385 | -1.473 | 79 | 66 | 14.15 |
| gen_300k | AFM-G | 2.8928 | 6.518 | -7.744 | 383 | 375 | 4.85 |
| gen_300k | DLM | 2.8572 | 9.732 | -0.000 | 0 | 0 | 0.08 |
| gen_400k | FM | 2.8363 | 11.064 | -0.000 | 0 | 0 | 0.15 |
| gen_400k | NC-90 | 2.8396 | 10.302 | -0.003 | 8 | 0 | 10.64 |
| gen_400k | AFM-A | 2.8521 | 10.322 | -1.775 | 92 | 79 | 13.99 |
| gen_400k | AFM-G | 2.8997 | 6.671 | -7.839 | 387 | 379 | 5.08 |
| gen_400k | DLM | 2.8581 | 9.702 | 0.000 | 0 | 0 | 0.06 |
| gen_500k | FM | 2.8374 | 10.968 | -0.000 | 0 | 0 | 0.12 |
| gen_500k | NC-90 | 2.8395 | 10.248 | -0.003 | 8 | 0 | 10.42 |
| gen_500k | AFM-A | 2.8518 | 10.331 | -1.648 | 82 | 70 | 14.15 |
| gen_500k | AFM-G | 2.9005 | 6.627 | -7.651 | 375 | 367 | 5.01 |
| gen_500k | DLM | 2.8582 | 9.656 | 0.000 | 0 | 0 | 0.07 |

Two counts are given for imaginary modes on purpose. `modes < -1e-3 THz` is the raw count and includes the three acoustic modes at Γ, which sit at −3e-3 THz because the force constants are float32. `modes < -0.1 THz` is the physical count. AFM-G is the only state with a genuine instability.

**ASR caveat.** For the multi-atom magnetic cells the sum rule violation before correction is 11-16 eV/Å² (FM: 0.23), because the force constants of a canted or layered cell are not cubic-symmetrisable and the one-representative-per-basis displacement scheme leaves a larger residual. It is absorbed by the on-site correction and the resulting Γ acoustic modes come out within 3e-3 THz of zero, but the absolute frequencies of those states carry a larger systematic uncertainty than the FM curve.

## 5b. 声速与弹性常数（由同一套声子谱微分得到）

Slope of the acoustic branches at small q, no extra force evaluations. `v = 0.6283 x d(nu)/dq` in km/s for nu in THz and q in 1/A; `C = rho v^2` with rho = 2 m_Fe / a^3.

| checkpoint | state | v_L[001] (km/s) | v_T[001] | v_L[111] | C_L (GPa) | C_T (GPa) | v_T < 0 |
|---|---|---|---|---|---|---|---|
| gen_110k | FM | 7.14 | 4.34 | 7.46 | 415 | 153 | - |
| gen_110k | NC-90 | 5.68 | 4.57 | 7.36 | 261 | 169 | - |
| gen_110k | AFM-A | 5.09 | 4.36 | 7.19 | 207 | 152 | - |
| gen_110k | AFM-G | 4.50 | -0.99 | 6.42 | 157 | 8 | yes |
| gen_110k | DLM | 5.51 | 4.35 | 7.02 | 241 | 150 | - |
| gen_200k | FM | 7.08 | 4.31 | 7.46 | 408 | 151 | - |
| gen_200k | NC-90 | 5.84 | 4.62 | 7.56 | 276 | 173 | - |
| gen_200k | AFM-A | 5.23 | 4.41 | 7.31 | 219 | 156 | - |
| gen_200k | AFM-G | 4.66 | -2.52 | 6.43 | 166 | 49 | yes |
| gen_200k | DLM | 5.57 | 4.43 | 7.15 | 246 | 156 | - |
| gen_300k | FM | 7.16 | 4.30 | 7.47 | 417 | 150 | - |
| gen_300k | NC-90 | 5.85 | 4.61 | 7.57 | 277 | 173 | - |
| gen_300k | AFM-A | 5.13 | 4.37 | 7.22 | 210 | 153 | - |
| gen_300k | AFM-G | 4.43 | -3.54 | 5.99 | 151 | 96 | yes |
| gen_300k | DLM | 5.52 | 4.40 | 7.10 | 242 | 154 | - |
| gen_400k | FM | 7.18 | 4.32 | 7.51 | 420 | 152 | - |
| gen_400k | NC-90 | 5.84 | 4.64 | 7.61 | 276 | 174 | - |
| gen_400k | AFM-A | 5.11 | 4.42 | 7.28 | 208 | 156 | - |
| gen_400k | AFM-G | 4.53 | -3.76 | 6.07 | 156 | 107 | yes |
| gen_400k | DLM | 5.44 | 4.44 | 7.11 | 235 | 157 | - |
| gen_500k | FM | 7.20 | 4.37 | 7.56 | 421 | 155 | - |
| gen_500k | NC-90 | 5.89 | 4.69 | 7.66 | 281 | 179 | - |
| gen_500k | AFM-A | 5.16 | 4.49 | 7.34 | 213 | 161 | - |
| gen_500k | AFM-G | 4.58 | -3.60 | 6.13 | 160 | 99 | yes |
| gen_500k | DLM | 5.49 | 4.48 | 7.14 | 239 | 159 | - |

Reference: alpha-Fe has v_L ~ 5.9 km/s, v_T ~ 3.2 km/s, C11 = 231 GPa, C44 = 117 GPa. A negative v_T marks either a genuine soft branch (AFM-G) or the ASR residual of a cell that cannot be cubic-symmetrised; it is not a physically softened mode in the latter case.

## 6. 磁–声耦合

Three independent measurements, because they answer different questions:
1. the mixed derivative `d²E/du dS` at one site — the linear magnon-phonon vertex, which must vanish at a collinear state (spin-rotation invariance);
2. the crossings of the magnon branch with the phonon branches — with a zero vertex they must be exact crossings, not avoided ones;
3. `d ln ω / d ln|S|` from rebuilding the whole dynamical matrix at two spin magnitudes — the magnetostrictive (exchange-striction) channel, which is the one that actually shifts the dispersion.

| checkpoint | C_transverse (eV/Å/μB) | C_longitudinal | noise floor | transverse resolved? | # crossings | max dlnω/dln m | dln a0/dln m |
|---|---|---|---|---|---|---|---|
| gen_110k | -7.95e-05 | -1.79e-04 | 2.68e-03 | no | 7 | 0.360 | 0.0367 |
| gen_200k | -1.89e-03 | -1.47e-03 | 2.34e-03 | no | 8 | 0.395 | 0.0372 |
| gen_300k | -9.34e-04 | +1.39e-03 | 2.09e-03 | no | 7 | 0.407 | 0.0392 |
| gen_400k | -8.54e-04 | -8.34e-04 | 1.49e-03 | no | 7 | 0.425 | 0.0407 |
| gen_500k | -2.82e-03 | -2.19e-04 | 3.07e-03 | no | 7 | 0.440 | 0.0419 |

The last two columns are the two halves of the same magnetoelastic coupling: how much the phonons stiffen (`dlnω/dln|S|`) and how much the lattice expands (`dln a0/dln|S|`) per fractional change of the moment. `dln a0/dln|S|` is measured on the 54-atom EOS cell used everywhere else for a0, so it is comparable with column 6 of the first table.

## 7. 这些数字不能说什么

* `J_n` are effective Fourier components of the model's *angular* energy functional inside a 6 Å cutoff, not hopping integrals. They depend on the moment used (`J/m²` is given for comparison with Heisenberg parameters).
* The magnon dispersion is the harmonic (LSWT) dispersion of those `J_n`. There is no magnon solver in the code, and no magnon lifetime, damping or magnon-magnon interaction is obtained.
* `Tc` comes from a 1 ps SLLG Langevin run per temperature at L = 6; it is a sampler-dependent estimate (the dt-dependence is a known open failure), not a precision critical temperature.
* The DLM phonons average over 2 configurations only; the configuration spread is not resolved by that.
* Nothing here says the 0.1 fs production step is validated for all materials or durations — that remains scoped as in the main report.
