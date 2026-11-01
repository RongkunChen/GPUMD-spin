"""Stage PV1 -- independent checks of the heat-current machinery.

Three experiments, each of which can fail on its own:

``cont``   energy-moment continuity.  In a periodic NVE run the extensive heat
           current must satisfy  d/dt sum_i r_i e_i = J  with e_i = e_i^pot +
           e_i^kin and r_i continuous (unwrapped).  Both sides are measured
           from the same trajectory: the left side by finite differences of the
           dump, the right side is what `dump_heat_flux` writes.  No analytic
           model is involved, so this tests the implementation and not a
           derivation.
``shift``  translation invariance.  The same run with every coordinate shifted
           by a constant vector must give the same J(t); a flux that depends on
           the origin of the cell is not a flux.
``gk``     Green-Kubo sampling: a 5 ps NVE trajectory with J dumped every 10
           steps, used for the correlation function, the correlation time, the
           running integral and its block error.

The `cont` run needs per-atom potential energy and unwrapped positions, which
`dump_xyz` provides (`potential`, `unwrapped_position`, `velocity`, `spin`,
`torque`).  Dumping every step for 128 atoms for 400 steps is ~20 MB, which is
acceptable; nothing larger is dumped at that rate.
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.path.join(ROOT, "evidence", "stage_PV1")
OUT = os.path.join(STAGE, "heat_audit")
CASES = os.path.join(OUT, "cases")
SYS = os.path.join(STAGE, "fe_lt", "systems")
GPUMD = os.path.join(ROOT, "final_delivery_20261031_174555", "win_staging", "gpumd.exe")
NEP = os.path.join(
    ROOT, "final_delivery_20261031_174555", "selected_models", "mask17_baseline", "nep.txt"
)
EQ = os.path.join(STAGE, "fe_lt", "cases_250K", "eq_1024_T250", "eq_last.xyz")
# the 128-atom equilibration is the only 128-atom frame that carries velocities
EQ128 = os.path.join(STAGE, "fe_lt", "cases_250K", "eqsz_fe128_ds_fm", "eq_last.xyz")


def run_case(name, xyz, lines, timeout=3600):
    d = os.path.join(CASES, name)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(NEP, os.path.join(d, "nep.txt"))
    shutil.copyfile(xyz, os.path.join(d, "model.xyz"))
    with open(os.path.join(d, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        p = subprocess.run([GPUMD], cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    print(f"  {name}: rc={p.returncode}" + (" GUARD" if "RUNAWAY GUARD" in txt else ""), flush=True)
    return p.returncode


def cont_case():
    """400 steps, dumped every step: the continuity test."""
    lines = [
        "potential nep.txt",
        "time_step 0.05 10",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 1",
        "dump_xyz 1 traj.xyz potential velocity unwrapped_position spin torque",
        "dump_heat_flux 1",
        "seed 71",
        "run 400",
    ]
    # 128 atoms keeps the every-step dump at ~40 MB.  The input MUST carry
    # velocities: a freshly replicated cell has none, and then the "dynamics"
    # is a frozen lattice with J identically zero (this happened on the first
    # attempt and is why the source is an equilibrated frame).
    run_case("cont_N128", EQ128, lines)


def shift_case():
    """Same as `cont` but every coordinate moved by a constant vector."""
    import numpy as np
    from math import floor

    src = EQ128
    dst = os.path.join(SYS, "fe128_ds_fm_shifted.xyz")
    with open(src) as f:
        lines = f.readlines()
    n = int(lines[0])
    head = lines[1]
    import re
    lat = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', head).group(1).split()])
    lat = lat.reshape(3, 3)
    shift = np.array([0.317, 0.191, 0.233])
    rows = [ln.split() for ln in lines[2 : 2 + n]]
    out = [f"{n}", head.rstrip("\n")]
    for r in rows:
        p = np.array([float(x) for x in r[1:4]]) + shift
        s = p @ np.linalg.inv(lat)
        s -= np.floor(s)
        p = s @ lat
        rr = list(r)
        rr[1:4] = [f"{x:.10f}" for x in p]
        out.append(" ".join(rr))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    lines = [
        "potential nep.txt",
        "time_step 0.05 10",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 10",
        "dump_heat_flux 1",
        "seed 72",
        "run 400",
    ]
    run_case("shift_N128", dst, lines)


def gk_case():
    """5 ps at 250 K, J every 10 steps."""
    lines = [
        "potential nep.txt",
        "time_step 0.05 100",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 100",
        "dump_heat_flux 10",
        "seed 81",
        "run 100000",
    ]
    run_case("gk_N1024", EQ, lines)


def hac_case():
    """Cross-check the shipped `compute_hac` against our own estimator.

    `compute_hac` refuses to run under NVE (it needs a target temperature), so
    this case uses a Langevin thermostat with tau = 1e6 steps: the friction and
    the noise are ~1e-6 of the physical values, i.e. the trajectory is NVE to
    within the sampling error, but `Integrate::temperature` is 250 K.

    Only the z column is compared: `compute_hac` correlates its five
    jx_in/jx_out/jy_in/jy_out/jz components separately and therefore drops the
    in/out cross term in x and y, while jz is the full z flux.
    """
    lines = [
        "potential nep.txt",
        "time_step 0.05 100",
        "ensemble nvt_lan 250 250 1000000",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 100",
        "dump_heat_flux 10",
        "compute_hac 10 2000 100",
        "seed 91",
        "run 20000",
    ]
    run_case("hac_N1024", EQ, lines)


def cont2_case():
    """Same as `cont` plus the per-atom virial and forces.

    Needed to separate "my energy-moment is wrong" from "the code's W_i is not
    the conduction flux": with the dumped virial the code's own phonon term
    sum_i W_i.v_i can be rebuilt exactly and compared both with the dumped J
    and with d/dt sum r e.
    """
    lines = [
        "potential nep.txt",
        "time_step 0.05 10",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 1",
        "dump_xyz 1 traj.xyz potential velocity unwrapped_position spin torque virial force",
        "dump_heat_flux 1",
        "seed 73",
        "run 400",
    ]
    run_case("cont2_N128", EQ128, lines)


def cont_nospin_case():
    """The same continuity run without spin dynamics.

    Decisive control: with no spin the flux is only sum_i W_i.v_i, and the
    energy-moment identity is the textbook pair-potential statement.  If this
    closes and the spin run does not, the discrepancy is in the spin channel;
    if neither closes, the test itself (or the phonon channel) is at fault.
    """
    lines = [
        "potential nep.txt",
        "time_step 0.05 10",
        "ensemble nve",
        "dump_thermo 1",
        "dump_xyz 1 traj.xyz potential velocity unwrapped_position virial force",
        "dump_heat_flux 1",
        "seed 74",
        "run 400",
    ]
    run_case("cont_nospin_N128", EQ128, lines)


def cont3_case():
    """Continuity test in a cell that is large enough for the identity to hold.

    d/dt sum_i r_i e_i = sum_i r_i e_i-dot is only the heat current when the
    interaction vectors really are r_i - r_j.  With a minimum-image cutoff the
    bond vector differs from r_i - r_j by a lattice vector, so the identity
    picks up an extra sum n.L j_ij that does NOT vanish in a small cell.  The
    128-atom cell is 9.70 x 9.75 x 14.98 A with a 6 A radial cutoff: L < 2 r_c
    in x and y, so the test cannot close there (measured: factor 18).  The
    1024-atom cell is 19.4 x 19.5 x 30.0 A, i.e. L > 2 r_c in every direction.
    """
    lines = [
        "potential nep.txt",
        "time_step 0.05 10",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        "dump_thermo 1",
        "dump_xyz 1 traj.xyz potential velocity unwrapped_position spin torque virial",
        "dump_heat_flux 1",
        "seed 75",
        "run 300",
    ]
    run_case("cont3_N1024", EQ, lines)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(CASES, exist_ok=True)
    if what in ("cont", "all"):
        cont_case()
    if what in ("shift", "all"):
        shift_case()
    if what in ("gk", "all"):
        gk_case()
    if what in ("cont2", "all"):
        cont2_case()
    if what in ("nospin", "all"):
        cont_nospin_case()
    if what in ("cont3", "all"):
        cont3_case()
    if what in ("hac", "all"):
        hac_case()
