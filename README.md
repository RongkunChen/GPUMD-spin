# GPUMD-spin V3

A spin-extended fork of [GPUMD](https://github.com/brucefan1983/GPUMD): a GPU
molecular dynamics package that trains neuroevolution potentials (NEP) carrying
magnetic spin degrees of freedom and runs finite-temperature spin dynamics with
them.

Two things are added on top of upstream GPUMD:

1. **Spin-aware NEP models.** The descriptor set is extended by five spin
   blocks (`Sex`, `San`, `Sdm`, `Sia`, `Smag`) and the trainer learns the
   magnetic torque $-\partial E/\partial\mathbf S_i$ in addition to energy,
   forces and virial.
2. **Spin-lattice dynamics in MD.** A stochastic Landau–Lifshitz–Gilbert
   (sLLG) integrator advances the spins while the lattice is integrated by
   velocity Verlet, coupled through a symmetric (Strang) split.

Version: see `VERSION`. Current release tag: `v3.1.0-r19`.

---

## 1. Introduction and features

### 1.1 What the code does

* **MD** with empirical and machine-learned potentials, on a single GPU.
* **NEP training** (`nep` executable) including the spin extension.
* **Spin dynamics**: stochastic LLG with a fixed spin length, coupled to the
  lattice; deterministic conservative propagation when damping and temperature
  are both zero.
* **Replica-exchange MD** (`gpumd_replica` executable).
* **Monte Carlo** (canonical, variance-constrained and semi-grand-canonical)
  mixed with MD, for NEP potentials.
* **Energy minimisation** (steepest descent, FIRE), with optional box
  relaxation.
* **Lattice dynamics**: force constants and phonon dispersion by finite
  displacement.
* **Transport**: Green–Kubo, HNEMD, spectral heat flux, modal analysis and
  linear-scaling quantum transport.
* **Thermal and mechanical scanning**: cohesive-energy curves, elastic
  constants, shock (MSST, piston walls), thermodynamic integration.

### 1.2 Potentials

The `potential` keyword accepts the following model types (names as written in
the model file header):

| Model | Notes |
|---|---|
| `nep4` | Standard NEP |
| `nep4_spin` | **Spin-extended NEP** (this fork) |
| `nep4_charge1`, `nep4_charge2`, `nep4_charge3` | NEP with charge / dipole / polarisability |
| `nep4_zbl`, `nep4_zbl_charge*`, `nep4_zbl_temperature` | NEP with ZBL repulsion at short range |
| `nep4_temperature` | Temperature-dependent NEP |
| `nep_ilp`, `tersoff_ilp`, `sw_ilp` | Interlayer potentials (Tersoff / Stillinger–Weber / NEP form) |
| `nnap`, `nnap_zbl` | Neural network atomic potential |
| `dp` | Deep Potential |
| `eam_zhou_2004`, `eam_dai_2006`, `adp` | EAM and angular-dependent potential |
| `tersoff_1988`, `tersoff_1989`, `tersoff_mini` | Tersoff |
| `lj` | Lennard-Jones |
| `fcp` | Force-constant potential |
| `average`, `observe` | Average of / observation over several loaded potentials |

DFT-D3 dispersion correction can be added on top with the `dftd3` keyword.

---

## 2. Theoretical background

### 2.1 Neuroevolution potential

For atom $i$ the NEP model builds a descriptor vector $\mathbf q_i$ from the
positions of its neighbours inside a cutoff, and maps it through a shallow
feed-forward network to a per-atom energy:

$$
U_i = \mathrm{ANN}(\mathbf q_i),\qquad U = \sum_i U_i
$$

The radial part is expanded on Chebyshev basis functions $g_n(r)$ and the
angular part on real spherical harmonics of degree $l$, both multiplied by a
smooth cutoff $f_c(r)$. In this fork the descriptor is extended by spin blocks,
see §2.3.

### 2.2 Spin dynamics

The spin equation solved in `src/spin/spin.cu` is the **Landau–Lifshitz form**
of the stochastic LLG equation (see `src/spin/spin.cuh`):

$$
\frac{d\mathbf S_i}{dt} = -\gamma\,\mathbf S_i\times\mathbf B_i
-\frac{\gamma\alpha}{|\mathbf S_i|}\,\mathbf S_i\times\big(\mathbf S_i\times\mathbf B_i\big)
$$

with the effective field

$$
\mathbf B_i = \mathbf w_i + \mathbf b_i(t) + \mathbf b_{\rm ext},
\qquad \mathbf w_i = -\frac{\partial E}{\partial\mathbf S_i}
$$

where:

* $\mathbf w_i$ is the magnetic torque produced by the potential, in eV/$\mu_B$;
* $\mathbf b_i(t)$ is the thermal field of the sLLG thermostat;
* $\mathbf b_{\rm ext}$ is the field set by `add_bfield`, in eV/$\mu_B$.

The noise is white in space and time with the fluctuation–dissipation
amplitude (`src/spin/spin.cuh`):

$$
\big\langle b_i^{a}(t)\,b_j^{b}(t')\big\rangle
= \frac{2\alpha k_B T}{\gamma\,(1+\alpha^2)\,|\mathbf S_i|}\,
\delta_{ij}\,\delta^{ab}\,\delta(t-t')
$$

discretised per step as

$$
\sigma_i = \sqrt{\frac{2\alpha k_B T}
{\gamma\,(1+\alpha^2)\,|\mathbf S_i|\,\Delta t}},
\qquad k_B = 8.617333262\times10^{-5}\ \mathrm{eV/K}
$$

The normalisation is **per atom** — it divides by $\|\mathbf S_i\|$, not by the
volume or the atom count.

**Units.** `gamma` is given in $\mu_B/(\mathrm{eV}\cdot\mathrm{fs})$. The
physical gyromagnetic ratio $g\mu_B/\hbar$ with $g=2$ equals
$3.0385\ \mu_B/(\mathrm{eV}\cdot\mathrm{fs})$, which is the value used in the
examples.

**Spin length.** $\|\mathbf S_i\|$ is conserved by every integration branch: the
geometric branch composes two isometries (a Rodrigues rotation and an analytic
Gilbert relaxation), and the midpoint branches converge to the implicit
midpoint rule, which is norm preserving.

**Integration branches** (`Spin::propagate` in `src/spin/spin.cu`):

| Condition | Scheme | Kernel |
|---|---|---|
| $\alpha = 0$ and $T = 0$ | Implicit midpoint, fixed-point iteration | `gpu_midpoint_update` |
| $T > 0$ and `scheme=midpoint` | Stochastic implicit midpoint | `gpu_midpoint_llg_update` |
| otherwise (default, `scheme=geometric`) | Rodrigues rotation + analytic relaxation, Strang-split | `spin_rotate`, `spin_relax` |

### 2.3 Spin descriptors

Every block sums over the neighbours $j$ of atom $i$ inside the cutoff, weighted
by the radial basis $g_n$ and the cutoff function $f_c$. The arguments are the
unit bond vector and its projections,

$$
\mathbf e_{ij}=\mathbf r_{ij}/r_{ij},\qquad
a=\mathbf S_i\cdot\mathbf e_{ij},\qquad
b=\mathbf S_j\cdot\mathbf e_{ij},\qquad
s=\|\mathbf S_i\|,
$$

which gives the five blocks of layout 3:

| Block | Definition | Physical order |
|---|---|---|
| `Sex` | $q^{\mathrm{Sex}}_{i,n}=\sum_j g_n f_c\,(\mathbf S_i\cdot\mathbf S_j)$ | exchange |
| `San` | $q^{\mathrm{San}}_{i,n}=\sum_j g_n f_c\,a\,b$ | anisotropic exchange |
| `Sdm` | $q^{\mathrm{Sdm}}_{i,n}=\sum_j g_n f_c\,\mathbf e_{ij}\cdot(\mathbf S_i\times\mathbf S_j)$ | Dzyaloshinskii–Moriya |
| `Sia` | $q^{\mathrm{Sia}}_{i,n}=\sum_j g_n f_c\,(\tfrac32 a^2-\tfrac12 s^2)$ | single-ion anisotropy |
| `Smag` | $q^{\mathrm{Smag}}_{i,n}=u^{\,n+1}$ | on-site moment magnitude |

`Smag` is a purely on-site term (no neighbour sum), built from $u=s\,m$ where
$m$ is the scaling factor given by `spin_mag_scale`. The current model layout is
version 3, written into `nep.txt` as `spin_layout 3 Sex San Sdm Sia Smag`.
Layout 2 used a deprecated `Senv` block in place of `Sia`.

**Expressive limits worth knowing before choosing a block mask.** `Sia` carries
spin rank 2 only, so it represents uniaxial/quadrupolar anisotropy but not a
cubic $l=4$ term of the form $S_x^4+S_y^4+S_z^4$. `Sdm` represents the
bond-parallel (screw) component of DMI only. `Smag` scales with $s$ and is not
differentiable at zero moment.

### 2.4 Spin–lattice coupling

The default coupling is the symmetric Strang split

$$
S(\Delta t/2)\;V(\Delta t)\;S(\Delta t/2)
$$

implemented in `Run::perform_run` (`src/main_gpumd/run.cu`): half spin step →
force → half kick plus drift → force → half kick → half spin step → force and
thermodynamic output. Each step therefore costs three force evaluations.

Conserved quantities: $\|\mathbf S_i\|$ always; total energy additionally when
$\alpha = 0$ and $T = 0$ (the conservative branch). With $T>0$ the sLLG
thermostat is a heat bath and the energy is not conserved.

### 2.5 Other physics implemented

* **Ensembles**: NVE, Berendsen/Nosé–Hoover/Langevin/BDP/BAOAB thermostats,
  Berendsen/SCR/MTK barostats, quantum thermal bath, path-integral and ring
  polymer MD, MSST shock, wall pistons, two-temperature model, and several
  thermodynamic-integration variants.
* **Transport**: the heat flux is decomposed into a phonon channel and a magnon
  channel $J_m=\sum_j A_j:\dot{\mathbf S}_j$ with
  $A_j=\sum_{i\neq j}(\mathbf r_i-\mathbf r_j)\otimes\partial U_i/\partial\mathbf S_j$
  (`src/measure/dump_heat_flux.cu`, `src/measure/compute_heat.cuh`).
* **Magnon spectra**: `compute_sed ... spin` uses the spin velocities; LSWT and
  frozen-spiral analysis are provided as Python tools under
  `tools/physics_validation/`.

---

## 3. Keyword (tag) reference for `run.in`

`run.in` is read line by line; each line starts with a keyword (referred to here
as a tag) followed by its parameters. Keywords may appear in any order subject
to the dependencies noted below.

### 3.1 Structure, potential and global control

| Tag | Parameters | Meaning |
|---|---|---|
| `replicate` | `<nx> <ny> <nz>` | Replicate the cell of `model.xyz`; must be the first effective command |
| `potential` | `<file> [<x\|y\|z>]` | Load a potential. The second argument selects the multi-GPU partition direction, or a second file for `dp`/`nnap`/`nep_ilp` |
| `time_step` | `<dt_fs> [<max_dist_per_step_A>]` | Time step in fs; optional adaptive-stepping distance limit |
| `run` | `<n_steps>` | Run MD; requires an `ensemble` defined earlier |
| `velocity` | `<T_initial> [seed <n>]` | Initialise velocities at a temperature |
| `seed` | `<int>` | Seed every random stream of the process |
| `correct_velocity` | `<interval> [<group_method>]` | Remove net linear/angular momentum periodically |
| `fix` | `<group_id>` or `<group_method> <group_id>` | Freeze a group |
| `move` | `<group_id> <vx> <vy> <vz>` | Impose a constant drift velocity (Å/fs) on a group |
| `change_box` | `<dxx> [<dyy> <dzz> [<dyz> <dxz> <dxy>]]` | Change the box (lengths in Å, or shear strains) and remap coordinates affinely |
| `deform` | `<rate> xx\|yy\|zz\|xy\|xz\|yz …` | Deform the box at a constant rate during the run |
| `minimize` | `sd <force_tol> <n_steps>` or `fire <force_tol> <n_steps> [<box_change> [<hydrostatic>]]` | Energy minimisation |
| `compute_phonon` | `<displacement>` | Force constants and phonon dispersion by finite displacement; requires `replicate` |
| `compute_cohesive` | `<start_factor> <end_factor> <deform_dir 0-6>` | Cohesive-energy curve |
| `compute_elastic` | `<strain>` | Elastic constants (strain ≤ 0.1) |
| `kspace` | `ewald` or `pppm [<spacing 0.2-2.0>]` | Long-range solver for NEP-Charge; may appear once |
| `dftd3` | `<xc_functional> <rc_potential> <rc_coord>` | DFT-D3 dispersion correction; may appear once |
| `deposit` | `<interval> <dir> <h_min> [<h_max>] atom <Elem> <n> <v> [<m>] …` or `… file <xyz> [<v>]` | Deposition simulation |

### 3.2 `ensemble` sub-types

| Tag | Parameters | Meaning |
|---|---|---|
| `ensemble nve` | — | Microcanonical |
| `ensemble nvt_ber` | `<T1> <T2> <tau_T>` | Berendsen thermostat |
| `ensemble npt_ber` | 6 / 10 / 16 params | Berendsen barostat |
| `ensemble nvt_nhc` | `<T1> <T2> <tau_T>` | Nosé–Hoover chain |
| `ensemble heat_nhc` | 5 params | NHC with two thermostats (NEMD) |
| `ensemble heat_nhc_power` | 5 params | NHC with constant power source |
| `ensemble nvt_lan` | `<T1> <T2> <tau_T>` | Langevin thermostat |
| `ensemble heat_lan` | 5 or 15 params | Langevin with heat source/sink groups |
| `ensemble nvt_bdp` | `<T1> <T2> <tau_T> [seed <n>]` | Bussi–Donadio–Parrinello |
| `ensemble heat_bdp` | 5 params | BDP with heat bath |
| `ensemble nvt_bao` | `<T1> <T2> <tau_T>` | BAOAB Langevin |
| `ensemble nvt_qtb` | 3 params + optional `f_max <v>` `N_f <n>` | Quantum thermal bath |
| `ensemble npt_scr` | 6 / 10 / 16 params | SCR barostat |
| `ensemble nvt_mttk` | `temp <T1> <T2> tperiod <n> [couple …]` | Martyna–Tobias–Klein thermostat |
| `ensemble npt_mttk` | as above + `iso/aniso/tri <p1> <p2> pperiod <n>` | MTK barostat |
| `ensemble nph_mttk` | as `npt_mttk` without `temp` | MTK constant enthalpy |
| `ensemble npt_qtb` | `temp …`, `iso/aniso/tri …`, `tperiod`, `pperiod`, `f_max`, `N_f` | QTB with barostat |
| `ensemble heat_ttm` | 17 positional params + optional `ttm_*` | Two-temperature model with source |
| `ensemble ttm` | 12 params + optional `ttm_*` | Two-temperature model |
| `ensemble heat_hybrid` | `<group_method> <group_id> <T> <tau> <dT> …` | Multiple thermostats (hybrid NEMD) |
| `ensemble rpmd` | `<n_beads>` | Ring polymer MD |
| `ensemble trpmd` | `<n_beads>` | Thermostatted ring polymer MD |
| `ensemble pimd` | 4 / 7 / 11 / 17 params [`eco <omega_max_cm1>`] | Path-integral MD |
| `ensemble pimd_scr` | 7 / 11 / 17 params [`eco <omega_max_cm1>`] | PIMD with SCR barostat |
| `ensemble msst` | `<v_shock>` + optional `qmass` `mu` `tscale` `p0` `v0` `e0` | Multi-scale shock technique |
| `ensemble ti` | `lambda` `temp` `tperiod` `spring <k> <r0>` | Thermodynamic integration |
| `ensemble ti_spring` | `tswitch` `tequil` `temp` `press` `tperiod` `spring` | TI with spring |
| `ensemble ti_rs` | `tperiod` `pperiod` `temp` `iso/aniso/tri` `tswitch` `tequil` | Reference-system TI |
| `ensemble ti_as` | as `ti_rs` | Adiabatic switching TI |
| `ensemble ti_liquid` | `tswitch` `tequil` `temp` `press` `tperiod` `sigmasqrd` `p` | TI for liquids |
| `ensemble ti_nep` | `tswitch` `tequil` `temp` `tperiod` | TI for NEP |
| `ensemble wall_piston` | `vp <v>` `thickness <d>` | Piston wall |
| `ensemble wall_mirror` | `vp <v>` | Reflecting wall |
| `ensemble wall_harmonic` | `vp <v>` `k <k>` | Harmonic wall |
| `ensemble nphug` | `tperiod` `pperiod` `iso/aniso/tri` `x/y/z` `p0` `v0` `e0` | Hugoniostat |

### 3.3 Spin tags (this fork)

| Tag | Parameters | Meaning |
|---|---|---|
| `spin_llg` | `<gamma> <alpha> [<T_K> [<seed>]]` or `gamma=… alpha=… [T=…] [seed=…] [scheme=geometric\|midpoint] [restart=<file>]` | Enable spin dynamics. `gamma` in $\mu_B/(\mathrm{eV\,fs})$, `alpha` dimensionless, `T` in K. `alpha=0` with `T=0` selects the conservative branch; `T>0` requires `alpha>0`. Requires a `spin` column in `model.xyz` |
| `spin_split` | `strang` (default) or `staggered` | Coupling split. `strang` is second order; `staggered` is the legacy first-order path |
| `spin_substeps` | `<n>` | Spin sub-steps per MD step. Takes effect only in the `staggered` path |
| `spin_torque_refresh` | `on` or `off` | Recompute the torque between sub-steps. Takes effect only in the `staggered` path |
| `add_bfield` | `<Bx> <By> <Bz> [<omega_THz>]` | External field in eV/$\mu_B$; `omega > 0` makes it sinusoidal (FMR / spin pumping). Requires `spin_llg` first |
| `dump_magnetization` | `<interval> [<file>] [warmup <n_samples>]` | Write $M_x, M_y, M_z, \|M\|$, variance, susceptibility and saturation, global and per type |

### 3.4 Output tags

| Tag | Parameters | Meaning |
|---|---|---|
| `dump_thermo` | `<interval>` | Thermodynamic output to `thermo.out` |
| `dump_restart` | `<interval>` | `restart.xyz`; with spin enabled also `spin_restart.bin` |
| `dump_xyz` | `<interval> <filename[*]> [group <gm> <gid>] [precision single\|double] [velocity\|force\|potential\|virial\|mass\|charge\|bec\|unwrapped_position\|spin\|torque\|group_labels …]` | Extended XYZ trajectory. A trailing `*` in the file name writes one file per frame |
| `dump_netcdf` | `<interval> <filename> [group …] [precision …] [compression …] [quantities …]` | NetCDF trajectory (build with `USE_NETCDF`) |
| `dump_cg` | `<interval> <grouping_method>` | Coarse-grained beads |
| `dump_beads` | `<interval> <has_velocity> <has_force>` | PIMD/RPMD beads |
| `dump_observer` | `observe\|average <thermo_interval> <exyz_interval> <has_velocity> <has_force>` | Energies/forces/virial from all loaded potentials |
| `dump_shock_nemd` | `interval <n> [bin_size <A>]` | Binned shock-NEMD profiles |
| `dump_dipole` | `<interval> <response_potential_file>` | Dipole moments |
| `dump_polarizability` | `<interval> <response_potential_file>` | Polarisability |
| `dump_heat_flux` | `<sample_interval>` or `<sample_interval> bin/1d <x\|y\|z> <nbin>` | Per-atom heat flux, with phonon and magnon channels written separately |

### 3.5 `compute_*` tags

| Tag | Parameters | Meaning |
|---|---|---|
| `compute` | `<group_method> <sample_interval> <output_interval> <quantities …>` | Spatio-temporal averages per group |
| `compute_chunk` | `<sample_interval> <output_interval> bin/1d <dim> <origin> <delta> temperature …` | Binned spatial profiles |
| `compute_dos` | `<sample_interval> <Nc> <omega_max_THz> [group …] [num_dos_points <n>]` | Phonon density of states |
| `compute_sdc` | `<sample_interval> <Nc> [group …]` | Self-diffusion coefficient |
| `compute_msd` | `<sample_interval> <Nc> [group …] [all_groups <gm>] [save_every <n>]` | Mean-square displacement |
| `compute_ic` | `<sample_interval> <Nc> <atom_type> <charge>` | Ionic conductivity |
| `compute_rdf` | `<rc> <num_bins> <sample_interval>` | Radial distribution function |
| `compute_adf` | `<sample_interval> <Nc> <rc> <num_bins> …` | Angular distribution function |
| `compute_orientorder` | `<interval> <cutoff\|nnn> <value> <ndegrees> [average 0/1] [wl 0/1] [wlhat 0/1]` | Steinhardt order parameters |
| `compute_angular_rdf` | `<interval> <Nc> <rc> <num_bins> …` | Angular-resolved RDF |
| `compute_sed` | `<sample_interval> <num_q> <x\|y\|z> [velocity\|spin] [0\|1]` | Spectral energy density; `spin` selects magnons and requires NVE |
| `compute_hac` | `<sample_interval> <Nc> <output_interval>` | Green–Kubo thermal conductivity |
| `compute_viscosity` | `<sample_interval> <Nc>` | Viscosity from stress autocorrelation |
| `compute_hnemd` | `<output_interval> <fe_x> <fe_y> <fe_z>` | HNEMD thermal conductivity (driving force in 1/Å) |
| `compute_hnemdec` | `<compute 0/1> <output_interval> <fe_x> <fe_y> <fe_z>` | HNEMD electrochemical variant |
| `compute_shc` | `<sample_interval> <Nc> <x\|y\|z> <num_omega> <max_omega> [group …]` | Spectral heat flux |
| `compute_gkma` | `<sample_interval> <first_mode> <last_mode> <bin_size\|f_bin_size> <value> [atom_range <b> <e>]` | Green–Kubo modal analysis |
| `compute_hnema` | `<sample_interval> <output_interval> <fe_x> <fe_y> <fe_z> <first_mode> <last_mode> <bin_size\|f_bin_size> <value> [atom_range …]` | HNEMD modal analysis |
| `compute_lsqt` | `<x\|y\|z> <num_moments> <num_E_points> <E_start> <E_end> <E_max> …` | Linear-scaling quantum transport |
| `compute_dpdt` | `<sample_interval>` | Dipole-moment time derivative (NEP-Charge only) |
| `compute_es` | `<sample_interval>` (must be 1) | Electrostatic energy and forces |
| `compute_extrapolation` | `[nep_file <f>] [asi_file <f>] [gamma_low <g>] [gamma_high <g>] [check_interval <n>] …` | NEP extrapolation monitoring |

### 3.6 Actions, external fields and sampling

| Tag | Parameters | Meaning |
|---|---|---|
| `add_force` | `<group_method> <group_id> <fx> <fy> <fz>` or `… <table_file>` | Constant or tabulated force on a group |
| `add_spring` | `ghost_com <gm> <gid> <vx> <vy> <vz> <stiff\|…> <k> <R0> <x0> <y0> <z0> [continue <id>]` | Spring force |
| `add_efield` | `<group_method> <group_id> <Ex> <Ey> <Ez> [charge\|bec]` or `… <table_file>` | Electric field |
| `add_random_force` | `<force_variance>` | Zero-mean random force |
| `electron_stop` | `<stopping_power_file>` | Electronic stopping |
| `mc` | `canonical <md_steps> <mc_steps> <T1> <T2> [group …]` or `sgc\|vcsgc … <num_types> <species> <mu\|phi> …` | Hybrid MC/MD (NEP only) |
| `active` | `<check_interval> <has_velocity> <has_force> <has_uncertainty> <threshold>` | On-the-fly active learning |
| `plumed` | `<input_file> <interval> <restart 0/1>` | PLUMED interface (build with `USE_PLUMED`) |
| `enhanced_sampling` | `<input_file>` | Enhanced sampling; output/start/cv/bias are given in the input file |

---

## 4. Usage

### 4.1 Building

Windows (PowerShell):

```bat
INSTALL.cmd            :: detect toolchain, build, run the smoke test
BUILD.cmd              :: build only
BUILD.cmd -NoSmoke     :: build without the smoke test
SMOKE_TEST.cmd         :: run the smoke test on an existing build
```

`BUILD.cmd` and `SMOKE_TEST.cmd` are thin wrappers that call `BUILD.ps1` and
`SMOKE_TEST.ps1` with the execution policy bypassed.

Linux / macOS:

```bash
./build.sh                     # builds gpumd, nep and gpumd_replica
CUDA_ARCH=-arch=sm_80 ./build.sh
```

The makefile lives in `src/`; its targets are

| Target | Executable | Purpose |
|---|---|---|
| `gpumd` | `src/gpumd` | Molecular dynamics |
| `nep` | `src/nep` | NEP training and prediction |
| `gpumd_replica` | `src/gpumd_replica` | Replica-exchange MD |
| `gnep` | — | Alternative NEP build |

The default CUDA architecture is `sm_60`; override it with `CUDA_ARCH`. The
optional NetCDF output and PLUMED interface require the corresponding build
flags (`USE_NETCDF`, `USE_PLUMED`).

### 4.2 Running

```bash
gpumd                     # reads run.in and model.xyz in the current directory
nep                       # reads nep.in and train.xyz  (training)
                          # or nep.in with prediction    (prediction)
gpumd_replica             # reads run.in with replica settings
```

### 4.3 `model.xyz`

Extended XYZ. The comment line carries the lattice, the column layout and the
periodicity:

```
16
Lattice="5.55088 0.06904 0.07774 0.0 5.79728 0.06187 0.0 0.0 5.51553" Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3 pbc="T T T"
Fe 0.09984 5.77222 0.09363 0.0 0.0 0.0 -1.60340 -1.51666 -0.67140
...
```

* Positions in Å, velocities in Å/fs, spin in $\mu_B$.
* The `spin:R:3` column is what makes a spin calculation possible; `spin_llg`
  and `dump_magnetization` both require it.
* A `group` column can be added to define grouping methods.

### 4.4 `nep.txt` (model file)

A spin-extended model starts with

```
nep4_spin 1 Fe
spin 1
spin_layout 3 Sex San Sdm Sia Smag
spin_mag_scale 0.25
spin_blocks Sex Smag
cutoff 6 3 84 14
n_max 4 4
basis_size 8 8
l_max 4 0 0
ANN 30 0
<weights>
```

* `spin_layout` names the five blocks in order; layout 3 uses `Sia`.
* `spin_blocks` records the mask actually used (written only when it differs
  from the full set). `Sex Smag` means mask 17.
* `spin_mag_scale` is the scale of the `Smag` block.

### 4.5 `nep.in` (training input)

```
type 1 Fe
cutoff 6.0 3.0
n_max 4 4
basis_size 8 8
l_max 4 0 0
neuron 30
spin 1
spin_blocks Sex Smag
spin_mag_scale 0.25
lambda_e 1.0
lambda_f 1.0
lambda_m 1.0
batch 32
population 20
generation 200
```

Spin-related parameters:

| Parameter | Meaning |
|---|---|
| `spin` | Per-type switch (one value, or one per type); must follow `type` |
| `spin_blocks` | Restrict the active blocks; disabled blocks are zeroed in energy, force and torque |
| `spin_mag_scale` | Scale of the `Smag` block; if omitted it is derived from the training data as $1/\max\|\mathbf S\|$ |
| `lambda_m` | Weight of the magnetic torque term $-\partial E/\partial\mathbf S$ in the loss |

The remaining loss weights are `lambda_e` (energy), `lambda_f` (force),
`lambda_v` (virial), `lambda_shear`, `lambda_q` (charge), `lambda_z` (BEC) and
`lambda_d`.

### 4.6 A complete `run.in`

```
potential nep.txt
time_step 0.5
ensemble nve
dump_thermo 20
dump_xyz 20 traj.xyz spin
spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=geometric
run 2000
```

---

## 5. Directory layout

```
src/
  main_gpumd/     gpumd entry point: run.in parsing, the MD step loop (run.cu)
  main_nep/       nep entry point: training (SNES), fitness, dataset, NEP kernels
  main_replica/   replica-exchange MD
  main_gnep/      alternative NEP build
  spin/           spin dynamics: Spin class, sLLG kernels, thermal field
  integrate/      velocity Verlet and every thermostat/barostat
  force/          potential implementations and neighbour lists
  measure/        dump_* and compute_* actions
  mc/             Monte Carlo (canonical, sgc, vcsgc)
  minimize/       steepest descent and FIRE
  model/          box, atoms, grouping
  phonon/         phonon post-processing
  utilities/      common definitions, descriptors, GPU helpers
  lib/            third-party sources
  makefile        build rules for gpumd / nep / gpumd_replica
docs/             theory notes, installation and validation documents
examples/         ready-to-run inputs (see §7)
models/           trained NEP models and training inputs
tests/            smoke, acceptance and physics test drivers
tools/            analysis and validation scripts
results/          recorded training runs and validation campaigns
scripts/          batch drivers
```

---

## 6. Implementation

### 6.1 MD call chain

```
main.cu : main()
  └─ Run::parse_run_in()                     read run.in, dispatch each tag
  │    ├─ "spin_llg"  → Spin::parse, Spin::initialize, atom.spin_dynamics = &spin
  │    ├─ "potential" → Force::parse
  │    └─ ... other tags
  └─ Run::perform_run()                      main loop
       for each step:
         A  spin.propagate(0.5*dt)                       S(dt/2)
         B  compute_force()                              forces at (r_n, S_n+1/2)
         C  integrate.compute1()                         half kick + drift
         D  compute_force()
         E  integrate.compute2()                         second half kick
         F  spin.propagate(0.5*dt)                       S(dt/2)
         G  compute_force(); integrate.find_thermo()     isochronous output state
         H  measure.end_of_step()
```

### 6.2 Spin propagation

```
Spin::propagate(dt, ...)                     spin.cu
  ├─ gpu_fill_thermal_field                  thermal field when T > 0
  ├─ propagate_conservative                  alpha == 0 && T == 0
  │    └─ gpu_midpoint_average / gpu_midpoint_update / gpu_midpoint_work
  ├─ propagate_stochastic_midpoint           T > 0 && scheme == midpoint
  │    └─ gpu_midpoint_llg_update
  ├─ gpu_llg_step                            default geometric scheme
  │    ├─ spin_rotate                        Rodrigues rotation
  │    └─ spin_relax                         analytic Gilbert relaxation
  └─ project_onto_norm_anchor                restore |S_i| after the stochastic branch
```

The field provider abstraction (`SpinFieldProvider` in `src/spin/spin.cuh`,
`NepSpinFieldProvider` in `spin_field_provider.cuh`) lets `spin.cu` be compiled
into both `gpumd` and `gpumd_replica` despite their different `Atom` layouts:
the spin code only sees raw device pointers, and the provider re-evaluates the
torque into a private scratch buffer during the midpoint iterations.

### 6.3 Torque pipeline

```
Force::compute(...)
  └─ NEP::compute
       ├─ find_descriptor_spin                       spin_descriptor_kernels.cuh
       ├─ ANN forward pass  →  ∂U/∂q
       ├─ find_force_spin                            forces, virial, torque, magnon tensor
       └─ convert_torque_to_float                    double accumulation → float output
```

Torques are accumulated in double precision and narrowed to `float` once per
step. The effective field used by the integrator is
$\mathbf w_i = -\partial E/\partial\mathbf S_i$.

### 6.4 Training call chain

```
main_nep/main.cu : main()
  ├─ Parameters::parse      "spin", "spin_blocks", "spin_mag_scale", "lambda_m"
  ├─ Fitness::compute       energy / force / virial / torque RMSE
  │    └─ Dataset::get_rmse_spin        compares predicted torque with reference
  ├─ SNES                   natural evolution strategy over the weights
  └─ Fitness::output        writes nep.txt including spin_layout / spin_mag_scale
```

---

## 7. Examples

| Directory | What it shows | How to run |
|---|---|---|
| `examples/spin_llg/` | Spin dynamics on 16 Fe atoms. Three variants: `run.in.conservative` ($\alpha=0$, $T=0$), `run.in.geometric`, `run.in.midpoint` | Copy one of the `run.in.*` files to `run.in` in a scratch directory together with `model.xyz` and `nep.txt`, then run `gpumd` |
| `examples/replica/` | Replica-exchange MD | `gpumd_replica` in the directory |
| `examples/restart/` | Restart continuation | `gpumd` with `dump_restart` |
| `examples/synthetic/` | Minimal synthetic system | `gpumd` |
| `examples/training/` | Training inputs `nep.in.mask17` and `nep.in.mask31` | `nep` with a training set |
| `examples/fe_bcc_ground_state/` | Ground-state scan for bcc Fe | `python scan.py` (see the directory README) |
| `examples/fe_curie_temperature/` | Curie-temperature scan | `python tc_scan.py` |
| `examples/fe_nve_stability/` | NVE energy-drift check | `python nve_drift.py` |

Minimal spin run:

```bash
mkdir -p /tmp/spin && cd /tmp/spin
cp <repo>/examples/spin_llg/{model.xyz,nep.txt} .
cp <repo>/examples/spin_llg/run.in.geometric run.in
<repo>/src/gpumd
```

Output: `thermo.out` every 20 steps and `traj.xyz` with the spin vectors.

---

## 8. License

This project is licensed under the **GNU General Public License v3.0**
(`LICENSES/GPL-3.0.txt`). Some components are under the GNU Lesser General
Public License v3.0 (`LICENSES/LGPL-3.0.txt`); see the individual file headers.

GPUMD is free software: you can redistribute it and/or modify it under the
terms of the GPL as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed
without any warranty.

---

## 9. Upstream project and acknowledgements

This repository is a fork of **GPUMD — Graphics Processing Units Molecular
Dynamics**, developed by Zheyong Fan and collaborators:

* Upstream repository: <https://github.com/brucefan1983/GPUMD>
* User manual: <https://gpumd.org/>
* Development manual: <https://gpumd.org/dev/>
* Tutorials: <https://github.com/brucefan1983/GPUMD-Tutorials>

The GPU infrastructure, the NEP implementation, the empirical potentials, the
integrators, the measurement framework and the example set all originate from
upstream. The spin extension — the spin descriptors, the magnetic torque, the
sLLG integrator and its coupling to the lattice — is developed in this fork.

If you use this code, please cite the upstream GPUMD papers. If you use the spin
extension, please contact <shiqian@ynu.edu.cn> or follow our recent work.
