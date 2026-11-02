/*
    S05A: concrete SpinFieldProvider -- the production NEP effective field.

    The conservative spin update (alpha = 0, T = 0) is the implicit midpoint
    rule, so it needs w = -dE/dS evaluated at the midpoint spins after *every*
    atom has moved.  This header supplies that field through the entry points
    the two executables already use for a force evaluation:

      gpumd          force.compute(box, pos, type, group, pot, frc, vir,
                                   vel, mass, position_image, spin, torque)
      gpumd_replica  force.set_spin(spin, torque);
                     force.compute(box, pos, type, group, pot, frc, vir)

    The two trees have different Force::compute overloads and different Atom
    layouts, so the two flavours are separate classes rather than one class with
    a branch: with C++14 both branches of an `if` would have to compile, and one
    of them never does in the other tree.

    FORCE is a template parameter (not simply Force&) so that the body of each
    class is only type-checked where it is instantiated: the main tree's Force
    has the 13-argument compute(), the replica tree's has set_spin() plus an
    8-argument compute(), and neither can be named in the other tree.

    Instantiate each class only in its own tree's translation unit.  Never name
    Atom, Box, Force or Group inside src/spin/spin.cu -- that isolation is what
    lets one spin.obj serve both executables (see the note at the top of
    spin.cuh).

    Scratch buffers: the potential/force/virial produced by a midpoint field
    evaluation belong to a state the run never reports, so they are written into
    private buffers and never into atom.potential_per_atom / force_per_atom /
    virial_per_atom.  potential_energy() therefore reports the *last*
    compute_field() call; call it at the state of interest.
*/

#pragma once

#include "spin.cuh"

#include <vector>

// gpumd: the field is passed straight to Force::compute, which does
// potentials[i]->set_spin(spin, torque) internally.
template <class ATOM, class FORCE>
class NepSpinFieldProvider : public SpinFieldProvider
{
public:
  NepSpinFieldProvider(
    ATOM& atom, FORCE& force, Box& box, const std::vector<Group>& group,
    int* position_image = nullptr)
    : atom_(atom),
      force_(force),
      box_(box),
      group_(group),
      position_image_(position_image)
  {
    potential_.resize(atom_.potential_per_atom.size(), 0.0);
    force_scratch_.resize(atom_.force_per_atom.size(), 0.0);
    virial_scratch_.resize(atom_.virial_per_atom.size(), 0.0);
  }

  void compute_field(
    const float* g_spin, float* g_field, const int number_of_atoms) override
  {
    (void)number_of_atoms;
    force_.compute(
      box_,
      atom_.position_per_atom,
      atom_.type,
      group_,
      potential_,
      force_scratch_,
      virial_scratch_,
      atom_.velocity_per_atom,
      atom_.mass,
      position_image_,
      g_spin,
      g_field);
    ++calls_;
  }

  double potential_energy(void)
  {
    std::vector<double> host(potential_.size(), 0.0);
    potential_.copy_to_host(host.data());
    double total = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
      total += host[i];
    }
    return total;
  }

  long long calls(void) const { return calls_; }

private:
  ATOM& atom_;
  FORCE& force_;
  Box& box_;
  const std::vector<Group>& group_;
  int* position_image_;
  GPU_Vector<double> potential_;
  GPU_Vector<double> force_scratch_;
  GPU_Vector<double> virial_scratch_;
  long long calls_ = 0;
};

// gpumd_replica: Force::compute takes no spin arguments there; the pointers are
// published with set_spin() first, exactly as the replica MD loop does.
template <class ATOM, class FORCE>
class NepSpinFieldProviderSetSpin : public SpinFieldProvider
{
public:
  NepSpinFieldProviderSetSpin(ATOM& atom, FORCE& force, Box& box,
                              std::vector<Group>& group)
    : atom_(atom), force_(force), box_(box), group_(group)
  {
    potential_.resize(atom_.potential_per_atom.size(), 0.0);
    force_scratch_.resize(atom_.force_per_atom.size(), 0.0);
    virial_scratch_.resize(atom_.virial_per_atom.size(), 0.0);
  }

  void compute_field(
    const float* g_spin, float* g_field, const int number_of_atoms) override
  {
    (void)number_of_atoms;
    force_.set_spin(g_spin, g_field);
    force_.compute(
      box_,
      atom_.position_per_atom,
      atom_.type,
      group_,
      potential_,
      force_scratch_,
      virial_scratch_);
    ++calls_;
  }

  double potential_energy(void)
  {
    std::vector<double> host(potential_.size(), 0.0);
    potential_.copy_to_host(host.data());
    double total = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
      total += host[i];
    }
    return total;
  }

  long long calls(void) const { return calls_; }

private:
  ATOM& atom_;
  FORCE& force_;
  Box& box_;
  std::vector<Group>& group_;
  GPU_Vector<double> potential_;
  GPU_Vector<double> force_scratch_;
  GPU_Vector<double> virial_scratch_;
  long long calls_ = 0;
};
