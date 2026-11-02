/*
 * Spin Route B descriptor core -- shared definitions.
 *
 * This header is the single place where the conventions of the native
 * equivariant spin descriptor (Route B, no pseudo atoms) are written down.
 * Everything that used to be duplicated between the three NEP
 * implementations -- the trainer in src/main_nep/, the MD kernel in
 * src/force/ and the replica fork in src/main_replica/force/ -- takes its
 * constants from here, so the three can no longer drift apart.
 *
 * The header deliberately has no dependencies beyond <cstddef>: it is
 * included by the trainer (which has neither `Box` nor `apply_mic`) as well
 * as by the two MD kernels.  The CUDA kernels themselves live in
 * `spin_descriptor_kernels.cuh`, which includes this file.
 */
#pragma once

#include <cstddef>

// sqrt(3 / (4 pi)).  Normalisation of a real l=1 spherical harmonic: it makes
// sum_j g_n(r) (r_hat . S_k) have the same scale as the other spin blocks, so
// that one ANN input scale covers all five.
#define SPIN_ENV_FACTOR 0.4886025119029199f

// String form of the same constant.  The JIT code generator in
// src/main_nep/nep_compile.cu emits a stand-alone .cu file that cannot include
// this header, so it writes the value out as text; taking the text from here
// keeps the generated code and the compiled code in step.  Only the legacy
// (layout <= 2) "Senv" block uses it.
#define SPIN_ENV_FACTOR_LITERAL "0.4886025119029199f"

// Spin block layout versions
//   1 = implicit legacy layout, no record in nep.txt
//   2 = explicit `spin_layout 2 Sex San Sdm Senv Smag`
//   3 = explicit `spin_layout 3 Sex San Sdm Sia  Smag`   <-- current
// Version 3 replaces the 4th block.  Senv = sum_j g_n (S_i . r_hat_ij) is
// linear in one spin, hence odd under time reversal: it makes the energy depend
// on the sign of the magnetisation, and its field sum_j g_n r_hat_ij is a
// lattice-only "permanent field".  It is inert only in a locally centrosymmetric
// environment, which is why the cubic test cells never exposed it.
// Sia is the single-ion anisotropy, i.e. S_i (x) S_i contracted with the
// traceless l=2 lattice quadrupole:
//     Q_i^{ab} = sum_j g_n(r_ij) [3 r_hat^a r_hat^b - delta^{ab}],
//     Sia      = 1/2 S_i^a S_i^b Q_i^{ab}.
// It is T- and P-even, invariant under a joint rotation of lattice and spins,
// identically zero in a locally cubic environment (a cubic crystal has no
// magnetocrystalline anisotropy) and non-zero for tetragonal / trigonal /
// surface sites.  Layout-2 models still load: paramb.spin_use_sia selects which
// block fills slot 3.  Derivation and symmetry table:
// docs/02-theory-and-implementation.md.
#define SPIN_LAYOUT_VERSION 3
#define SPIN_LAYOUT_VERSION_SENV 2
#define SPIN_LAYOUT_VERSION_IMPLICIT 1

// Number of spin descriptor blocks: Sex, San, Sdm, (Sia | Senv), Smag.
#define SPIN_NUM_BLOCKS 5

// Index of the 4th slot, the one whose meaning depends on the layout version.
#define SPIN_BLOCK_SLOT_REPLACED 3

// Canonical block names.  Spelled out as macros (rather than as a
// std::vector<std::string>) so that this header keeps its "no dependency
// beyond <cstddef>" property and can be consumed by the MD builds, the trainer
// and the JIT generator alike.
#define SPIN_BLOCK_NAME_SEX "Sex"
#define SPIN_BLOCK_NAME_SAN "San"
#define SPIN_BLOCK_NAME_SDM "Sdm"
#define SPIN_BLOCK_NAME_SENV "Senv"
#define SPIN_BLOCK_NAME_SIA "Sia"
#define SPIN_BLOCK_NAME_SMAG "Smag"

// S06: declared symmetry and capability of every spin block.
// The five blocks are NOT five interchangeable invariants.  Each one transforms
// in its own way, and two of them are identically zero for entire classes of
// configurations.  Before S06 the only place that said anything about this was
// the comment above the layout version, and it said nothing at all about Sdm.
// The table below is the single declaration the trainer, the MD kernels and the
// verification tools read; `tools/spin_validation/spin_symmetry_table_stage06`
// prints it and the S06 checker compares it with what the kernels actually do.
// Conventions.  A block value q_i is called
//   P-even / P-odd   if q_i is multiplied by +1 / -1 when the *configuration*
//                    (lattice AND spins) is inverted about the site: r -> -r,
//                    S -> S (the spin is an axial vector, so inversion does not
//                    flip it).  A P-odd scalar is a pseudoscalar.
//   T-even / T-odd   if q_i is multiplied by +1 / -1 under S -> -S.
//   spin-rotation    "invariant" if q_i does not change when every spin is
//                    rotated by the same R while the lattice is held fixed.
//                    A model built only from such blocks is the no-SOC case.
//                    "joint only" if q_i is invariant only when the lattice is
//                    rotated by the same R -- the SOC case, where the spin and
//                    the lattice are locked together.  A model containing a
//                    "joint only" block is NOT spin-rotation invariant, so it
//                    cannot describe a Hamiltonian that is.
//   spin rank        the degree in S_i of the block (0 = |S| only, 1 = linear,
//                    2 = bilinear).
// Reading of the table, block by block:
//   Sex   q = sum_j g_n (S_i . S_j)          P+ T+ rank 2, spin-rotation invariant
//         Isotropic exchange.  The only *bilinear* block that survives without
//         SOC, and the only one that a no-SOC dataset can legitimately use
//         together with Smag.
//   San   q = sum_j g_n (S_i.r)(S_j.r)       P+ T+ rank 2, joint only
//         The r.r exchange anisotropy.  Forbidden without SOC: it needs the
//         bond direction to define a preferred axis in spin space.
//   Sdm   q = sum_j g_n r.(S_i x S_j)        P- T+ rank 2, joint only
//         A PSEUDOSCALAR.  This is the bond-parallel Dzyaloshinskii-Moriya
//         term: the Moriya vector is constrained to D_ij = D(d_ij) r_ij, one
//         scalar per bond.  Consequences that the code must not paper over:
//           * it is P-odd, so it can never be called a general invariant.  A
//             model that contains it is not invariant under inversion of the
//             configuration unless the trained function happens to be even in
//             this component, which nothing enforces;
//           * it is identically zero for every configuration that is itself
//             inversion-symmetric (centrosymmetric lattice AND inversion-
//             symmetric spin arrangement, e.g. collinear).  For a centro-
//             symmetric lattice with a NON-symmetric spin texture it is not
//             zero -- the texture breaks the inversion, not the lattice;
//           * it cannot represent a general D_ij: the out-of-plane component
//             of an interfacial DMI, and the bond-perpendicular components of
//             a C3v/Cnv bond, are outside its span.  Only the bond-parallel
//             (spiral) component is representable;
//           * it is a chiral order parameter, so it belongs in a model only
//             when the target Hamiltonian breaks P.
//         Reference: this is the textbook "spin-chirality" coupling, allowed
//         exactly when the bond has no inversion centre.
//   Sia   q = sum_j g_n (1.5 a^2 - 0.5 |S_i|^2), a = S_i.r_ij
//         = 1/2 S_i^a S_i^b Q_i^{ab},  Q_i^{ab} = sum_j g_n (3 r^a r^b - d^{ab})
//                                           P+ T+ rank 2, joint only
//         Single-ion anisotropy of RANK 2 only: the traceless lattice
//         quadrupole contracted with the spin quadrupole.  It spans the
//         uniaxial / quadrupolar anisotropy of a site with lower-than-cubic
//         symmetry.  It is identically zero when the site is locally cubic
//         (every shell gives Q = 0 shell by shell), and every product of its
//         channels is then zero as well, so the rank-4 cubic anisotropy
//         K1 (Sx^4 + Sy^4 + Sz^4) -- the leading magnetocrystalline term of
//         bcc Fe -- is NOT representable by this block.  That is a capability
//         limit of the descriptor, not a training problem.
//   Senv  q = sum_j g_n (S_i . r_ij)         P- T- rank 1, joint only
//         DEPRECATED.  It is odd under time reversal, so an energy containing
//         it depends on the sign of the magnetisation, which no time-reversal-
//         symmetric Hamiltonian does; the lattice part sum_j g_n r_ij acts as
//         a permanent field.  It is retained only so that a layout-1/2 model
//         keeps evaluating exactly the function it was trained with.  New
//         models must not select it: layout 3 puts Sia in the same slot.
//   Smag  q_n = u^{n+1}, u = |S_i| * spin_mag_scale
//                                           P+ T+ rank 0, spin-rotation invariant
//         The |S| channel that carries the longitudinal magnetic force.
//         Continuous in S everywhere, but the n = 0 channel is proportional to
//         |S| and therefore its gradient S/|S| has no limit at |S| = 0: the
//         energy has a cone there.  Two consequences that the code must keep
//         straight:
//           * the torque is the exact derivative dq_n/dS = (n+1) u^n scale S/|S|
//             for every |S| > 0, and the kernels compute exactly that (see
//             SPIN_MAG_MIN_MOMENT for how the unit direction is formed when
//             |S| is too small to divide by);
//           * at |S| = 0 the derivative does not exist.  The kernels write 0
//             there, which is a convention at a non-differentiable point, not
//             a derivative, and it is the only place where the reported torque
//             is not -dE/dS.  A species whose moment can reach zero must not
//             carry this block.
// A model that mixes "spin-rotation invariant" and "joint only" blocks asserts
// that the spin-lattice coupling is SOC-like.  The Fe dataset of this project
// has no recorded SOC treatment and no recorded DFT provenance for its labels
// (UNKNOWN), so the choice is a modelling assumption, recorded here rather than
// hidden in a training script.
#define SPIN_DESCRIPTOR_SYMMETRY_VERSION 1

// Block indices, identical to the offset units of spin_block_offset().
#define SPIN_BLOCK_INDEX_SEX 0
#define SPIN_BLOCK_INDEX_SAN 1
#define SPIN_BLOCK_INDEX_SDM 2
#define SPIN_BLOCK_INDEX_SLOT4 3
#define SPIN_BLOCK_INDEX_SMAG 4

#define SPIN_PARITY_EVEN 1
#define SPIN_PARITY_ODD (-1)

// Magnitude below which the Smag torque forms the unit direction S/|S| by a
// scale-invariant normalisation instead of dividing the product by |S|.
// It is NOT a cutoff and NOT a floor in the denominator: the torque is the
// analytic derivative on both sides of it, and the two branches agree exactly
// at the threshold.  Above it the code keeps the original expression
// (c = .../|S|, then c*S), so the value is bit-identical to the build before
// S06; below it 1/|S| is either inaccurate or not representable in float32
// while the direction is perfectly well defined, so the direction is computed
// from a component-scaled vector.  Chosen far below any physical moment
// (|S| ~ 2 in the Fe data).
#define SPIN_MAG_MIN_MOMENT 1.0e-8f

// Text form of the same number.  The JIT generator writes a stand-alone .cu
// that cannot include this header (the specialized kernels deliberately do not
// depend on it), so the value is emitted as text, exactly like
// SPIN_ENV_FACTOR_LITERAL.  One number, two spellings, one place to change it.
#define SPIN_MAG_MIN_MOMENT_LITERAL "1.0e-8f"

#if defined(__CUDACC__)
#define SPIN_DESCRIPTOR_HD __host__ __device__
#else
#define SPIN_DESCRIPTOR_HD
#endif

// Which block occupies slot 4 for a given layout version: Sia from layout 3 on,
// the deprecated Senv before that.
SPIN_DESCRIPTOR_HD static inline int spin_slot4_is_sia(const int layout_version)
{
  return layout_version >= SPIN_LAYOUT_VERSION;
}

// F26: descriptor-block ablation.
// All blocks on by default (mask 31).  A clear bit makes that block contribute
// nothing: its descriptor values are written as zero AND its prefactor fp_* is
// zeroed.  Zeroing only one of the two silently pollutes either the forces
// (q = 0, fp != 0) or the training targets (fp = 0, q != 0).  The mask is
// written to nep.txt (only when not all-on) so that gpumd evaluates exactly the
// function the trainer trained.
#define SPIN_BLOCK_BIT_SEX   1
#define SPIN_BLOCK_BIT_SAN   2
#define SPIN_BLOCK_BIT_SDM   4
#define SPIN_BLOCK_BIT_SLOT4 8    // Sia (layout 3) or Senv (layout <= 2)
#define SPIN_BLOCK_BIT_SMAG  16
#define SPIN_BLOCK_MASK_ALL  31

// Number of spin descriptor components for a given n_max_angular.  Each block
// contributes n_max_angular + 1 radial channels.
static inline int spin_descriptor_dim(const int n_max_angular)
{
  return SPIN_NUM_BLOCKS * (n_max_angular + 1);
}

// Offset of block `b` (0 = Sex, 1 = San, 2 = Sdm, 3 = Sia|Senv, 4 = Smag)
// inside the per-atom spin descriptor, in units of (n_max_angular + 1) channels.
static inline int spin_block_offset(const int block, const int n_max_angular)
{
  return block * (n_max_angular + 1);
}

// Linear index of the descriptor component (block b, radial channel n) of atom
// n1 in the `N x dim_spin` descriptor array, which is stored atom-major.
static inline std::size_t spin_descriptor_index(
  const int n1, const int N, const int block, const int n, const int n_max_angular)
{
  return static_cast<std::size_t>(n1) +
         static_cast<std::size_t>(N) * (spin_block_offset(block, n_max_angular) + n);
}

// Index into the type-pair Chebyshev coefficient table, shared by the two MD
// kernels.  The replica fork used to inline this expression while src/force/
// called get_c_index() from nep_utilities.cuh; both are the same formula, and
// having it here means a change to the table layout is made once.
//   This is the same formula as get_c_index() in nep_utilities.cuh with
//   offset = num_c_radial; the replica fork used to inline it because its copy
//   of nep_utilities.cuh does not provide get_c_index().
template <typename ParaMB>
static __host__ __device__ __forceinline__ int spin_c_index(
  const ParaMB& paramb, const int t1, const int t2, const int n, const int k)
{
  return paramb.num_c_radial +
         (t1 * paramb.num_types + t2) *
           ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1)) +
         n * (paramb.basis_size_angular + 1) + k;
}

// S06: the declared table, as code.
// Every query takes the block index (0..4, i.e. the spin_block_offset() unit)
// and the layout version, because slot 4 means Sia or the deprecated Senv
// depending on it.  The functions are pure and dependency-free so that the MD
// kernels, the trainer and a host-side table printer can all use the same
// source of truth.

// Name of the block: "Sia" or "Senv" in slot 4, the fixed name elsewhere.
SPIN_DESCRIPTOR_HD static inline const char* spin_block_name(
  const int block, const int layout_version)
{
  if (block == SPIN_BLOCK_INDEX_SEX) return SPIN_BLOCK_NAME_SEX;
  if (block == SPIN_BLOCK_INDEX_SAN) return SPIN_BLOCK_NAME_SAN;
  if (block == SPIN_BLOCK_INDEX_SDM) return SPIN_BLOCK_NAME_SDM;
  if (block == SPIN_BLOCK_INDEX_SLOT4) {
    return spin_slot4_is_sia(layout_version) ? SPIN_BLOCK_NAME_SIA
                                             : SPIN_BLOCK_NAME_SENV;
  }
  return SPIN_BLOCK_NAME_SMAG;
}

// +1 if the block is even under inversion of the configuration, -1 if odd.
SPIN_DESCRIPTOR_HD static inline int spin_block_parity_inversion(
  const int block, const int layout_version)
{
  // Sdm = r.(S_i x S_j) is a pseudoscalar: r flips, the cross product of two
  // axial vectors does not.  Senv = S_i.r flips for the same reason.  Sia and
  // San are even because they are quadratic in r (San: two odd factors).
  if (block == SPIN_BLOCK_INDEX_SDM) return SPIN_PARITY_ODD;
  if (block == SPIN_BLOCK_INDEX_SLOT4 && !spin_slot4_is_sia(layout_version)) {
    return SPIN_PARITY_ODD;
  }
  return SPIN_PARITY_EVEN;
}

// +1 if the block is even under S -> -S, -1 if odd.
SPIN_DESCRIPTOR_HD static inline int spin_block_parity_time_reversal(
  const int block, const int layout_version)
{
  // Only the deprecated linear-in-Senv block is odd; every other block is
  // either quadratic in S (Sex, San, Sdm, Sia) or a function of |S| (Smag).
  if (block == SPIN_BLOCK_INDEX_SLOT4 && !spin_slot4_is_sia(layout_version)) {
    return SPIN_PARITY_ODD;
  }
  return SPIN_PARITY_EVEN;
}

// 1 if the block survives an independent global rotation of the spins with the
// lattice held fixed (the no-SOC case), 0 if it is invariant only under a joint
// rotation of lattice and spins (the SOC case).
SPIN_DESCRIPTOR_HD static inline int spin_block_invariant_under_spin_rotation(
  const int block, const int layout_version)
{
  (void)layout_version;
  return (block == SPIN_BLOCK_INDEX_SEX || block == SPIN_BLOCK_INDEX_SMAG) ? 1 : 0;
}

// Degree in S_i: 2 = bilinear in the moments, 1 = linear, 0 = |S| only.
SPIN_DESCRIPTOR_HD static inline int spin_block_spin_rank(
  const int block, const int layout_version)
{
  if (block == SPIN_BLOCK_INDEX_SMAG) return 0;
  if (block == SPIN_BLOCK_INDEX_SLOT4 && !spin_slot4_is_sia(layout_version)) {
    return 1;
  }
  return 2;
}

// 1 if the block is identically zero for any configuration that is invariant
// under inversion about the site (centrosymmetric lattice AND inversion-
// symmetric spin arrangement).  Note this is a statement about a symmetric
// configuration, not about the lattice alone.
SPIN_DESCRIPTOR_HD static inline int spin_block_vanishes_if_inversion_symmetric(
  const int block, const int layout_version)
{
  return spin_block_parity_inversion(block, layout_version) == SPIN_PARITY_ODD ? 1 : 0;
}

// 1 if the block is identically zero for a locally cubic environment, for every
// spin configuration (the lattice quadrupole and the permanent field both
// vanish shell by shell).
SPIN_DESCRIPTOR_HD static inline int spin_block_vanishes_in_cubic_environment(
  const int block, const int layout_version)
{
  if (block == SPIN_BLOCK_INDEX_SLOT4) return 1;  // Sia quadrupole, or Senv field
  return 0;
}

// 1 if every channel of the block is a smooth function of S everywhere,
// including |S| = 0.  Smag is 0: its n = 0 channel is proportional to |S|, which
// is continuous but not differentiable at the origin, so its torque is defined
// only in the limit and the kernels floor 1/|S| there.
SPIN_DESCRIPTOR_HD static inline int spin_block_analytic_at_zero_moment(
  const int block, const int layout_version)
{
  (void)layout_version;
  return block == SPIN_BLOCK_INDEX_SMAG ? 0 : 1;
}

// 1 for the blocks that exist only because the spin-lattice coupling breaks
// spin-rotation invariance, i.e. the ones a no-SOC dataset must mask off.
SPIN_DESCRIPTOR_HD static inline int spin_block_requires_soc(
  const int block, const int layout_version)
{
  return spin_block_invariant_under_spin_rotation(block, layout_version) ? 0 : 1;
}

// 1 for the legacy block that must not be used by a new model.
SPIN_DESCRIPTOR_HD static inline int spin_block_is_deprecated(
  const int block, const int layout_version)
{
  return (block == SPIN_BLOCK_INDEX_SLOT4 && !spin_slot4_is_sia(layout_version))
           ? 1 : 0;
}
