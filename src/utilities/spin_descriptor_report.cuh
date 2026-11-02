/*
 * S06: the human-readable form of the spin descriptor declaration.
 *
 * utilities/spin_descriptor.cuh holds the table as pure functions and keeps its
 * "no dependency beyond <cstddef>" property so the MD kernels and the trainer
 * can both use it.  Printing needs <cstdio>, so it lives here instead, in one
 * place, and is called by the MD entry point (force/nep.cu) and by the trainer
 * (main_nep/parameters.cu).  Two programs, one statement: a model cannot be
 * described differently depending on who opened it.
 */
#pragma once

#include "spin_descriptor.cuh"

#include <cstdio>

// One line per block of the active layout, then the consequences that a user
// has to know about: which blocks are dead for this geometry class, which one
// is deprecated, and whether the model asserts SOC.
// `mask` is the F26 ablation mask actually in force; a block that is switched
// off is reported as "off" rather than silently omitted, because "trained
// without this block" and "trained with it and it came out zero" are different
// statements about the model.
static inline void spin_report_declaration(
  const int layout_version, const int mask, std::FILE* out = stdout)
{
  if (out == nullptr) return;
  std::fprintf(out,
               "    spin descriptor declaration (symmetry version %d, layout %d)\n",
               SPIN_DESCRIPTOR_SYMMETRY_VERSION, layout_version);
  std::fprintf(out,
               "      %-5s %-3s %-3s %-14s %-5s %-9s %-9s %s\n",
               "block", "P", "T", "spin rotation", "rank",
               "inv.sym.", "cubic", "at |S|=0");
  int on_rotation_invariant = 0;
  int on_joint_only = 0;
  for (int b = 0; b < SPIN_NUM_BLOCKS; ++b) {
    const char* name = spin_block_name(b, layout_version);
    const int bit = 1 << b;
    const bool on = (mask & bit) != 0;
    std::fprintf(out, "      %-5s %-3s %-3s %-14s %-5d %-9s %-9s %s%s\n",
                 name,
                 spin_block_parity_inversion(b, layout_version) > 0 ? "+" : "-",
                 spin_block_parity_time_reversal(b, layout_version) > 0 ? "+" : "-",
                 spin_block_invariant_under_spin_rotation(b, layout_version)
                   ? "invariant" : "joint only",
                 spin_block_spin_rank(b, layout_version),
                 spin_block_vanishes_if_inversion_symmetric(b, layout_version)
                   ? "zero" : "-",
                 spin_block_vanishes_in_cubic_environment(b, layout_version)
                   ? "zero" : "-",
                 spin_block_analytic_at_zero_moment(b, layout_version)
                   ? "smooth" : "kink",
                 on ? "" : "   [off]");
    if (on) {
      if (spin_block_invariant_under_spin_rotation(b, layout_version)) {
        ++on_rotation_invariant;
      } else {
        ++on_joint_only;
      }
    }
  }
  if (mask & SPIN_BLOCK_BIT_SDM) {
    std::fprintf(out,
      "      Sdm is a PSEUDOSCALAR (chirality / bond-parallel DMI): its energy\n"
      "      term is odd under inversion of the configuration.  It is identically\n"
      "      zero only for configurations that are themselves inversion-symmetric\n"
      "      (centrosymmetric lattice AND symmetric spin arrangement), and it can\n"
      "      represent only the bond-parallel Moriya component.  It is a legal\n"
      "      descriptor for a P-broken target and is not a general invariant.\n");
  }
  if ((mask & SPIN_BLOCK_BIT_SLOT4) && !spin_slot4_is_sia(layout_version)) {
    std::fprintf(out,
      "      WARNING: the model uses the DEPRECATED Senv block (layout %d).\n"
      "      Senv is T-odd, so the energy depends on the sign of the\n"
      "      magnetisation, which no time-reversal-symmetric Hamiltonian does.\n"
      "      It is evaluated exactly as trained; do not build a new model on it.\n",
      layout_version);
  }
  if ((mask & SPIN_BLOCK_BIT_SLOT4) && spin_slot4_is_sia(layout_version)) {
    std::fprintf(out,
      "      Sia spans RANK-2 single-ion anisotropy only.  It is identically zero\n"
      "      for a locally cubic site, and so is every product of its channels;\n"
      "      the rank-4 cubic term K1*(Sx^4+Sy^4+Sz^4) is not representable.\n");
  }
  if (mask & SPIN_BLOCK_BIT_SMAG) {
    std::fprintf(out,
      "      Smag is continuous but not differentiable at |S| = 0: its n = 0\n"
      "      channel is linear in |S|, so the torque there is a limit, not a\n"
      "      derivative.  The kernels evaluate the exact derivative for every\n"
      "      |S| > 0 and write 0 only at |S| = 0 (a convention at a point where\n"
      "      no derivative exists).  The unit direction switches to a\n"
      "      scale-invariant normalisation below |S| = %s.\n",
      SPIN_MAG_MIN_MOMENT_LITERAL);
  }
  if (on_rotation_invariant && on_joint_only) {
    std::fprintf(out,
      "      This model mixes spin-rotation-invariant and joint-only blocks, so\n"
      "      it asserts an SOC-like spin-lattice coupling: the energy is invariant\n"
      "      under a joint rotation of lattice and spins, not under a rotation of\n"
      "      the spins alone.  A no-SOC dataset must use Sex|Smag only (mask 17).\n");
  } else if (on_joint_only == 0) {
    std::fprintf(out,
      "      Every enabled block is invariant under an independent global rotation\n"
      "      of the spins: the model is in the no-SOC class.\n");
  } else if (on_rotation_invariant == 0) {
    std::fprintf(out,
      "      Every enabled block is invariant only under a joint rotation of the\n"
      "      lattice and the spins: the model is in the SOC class.\n");
  }
  std::fflush(out);
}
