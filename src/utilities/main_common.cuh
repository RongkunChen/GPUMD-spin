/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

void print_compile_information(void);
void print_gpu_information(void);

/*
 * Global PRNG seed, set by the `seed <n>` command in run.in.
 *
 * GPUMD seeds its random streams in two different ways:
 *
 *   - the curand-based thermostats (nvt_lan, nvt_bao, npt_qtb, ...) call the
 *     host `rand()`, which `print_compile_information()` seeds from the wall
 *     clock in a release build;
 *   - some host-side thermostats (nvt_bdp, npt_scr, pimd) own a
 *     `std::mt19937` that is likewise seeded from the wall clock.
 *
 * Neither is reproducible unless the user can pin the seed.  `nvt_bdp` gained
 * its own `seed` keyword earlier; this is the general mechanism for the rest:
 * `seed <n>` in run.in calls `srand(n)` (so every later `rand()` is fixed) and
 * records `n` so the host-side generators can pick it up when they were not
 * given a seed of their own.  -1 means "the user did not ask", in which case
 * the historical wall-clock behaviour is kept.
 */
void set_user_prng_seed(int seed);

// The seed given by `seed <n>`, or -1 if the command was not used.
int user_prng_seed(void);
