/* nep_precision.h -- unified numeric output precision for the NEP trainer (main_nep).
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Before this policy existed, the trainer printed the same class of quantity with
 * different precisions depending on which module happened to emit it:
 *
 *   - structure-level NEP predictions used %.12g but the DFT reference sitting in
 *     the very same row used %g (6 significant digits);
 *   - atomic force / magnetic-torque files used %g everywhere, i.e. the spin
 *     channel -- the quantity this project validates to 1e-9 -- was written with
 *     the fewest digits of anything the trainer produces;
 *   - nep.txt and nep.restart used %15.7e (8 significant digits) while the JIT
 *     compiler embedded the same numbers with std::setprecision(9), so the
 *     on-disk model and the in-memory model were not bit-identical;
 *   - loss columns used fixed-point %.5f with three different field widths
 *     (9 / 11 / 13), which silently prints 0.00000 for any loss below 5e-6.
 *
 * A float32 round trip needs 9 significant decimal digits; 8 is not enough.
 * Anything written to disk and read back must therefore use NEP_FMT_PARAM.
 *
 * NEP_FMT_PARAM is "%.9e": nine digits AFTER the decimal point, i.e. ten
 * significant digits -- one more than the minimum, and exactly what the JIT
 * compiler already embeds via (std::scientific << std::setprecision(9)),
 * which also means nine digits after the point.  nep.txt and the JIT kernel
 * now carry bit-identical constants.
 *
 * THE FOUR CLASSES
 * ----------------
 *   P1 NEP_FMT_PARAM  persisted model parameters  (nep.txt, nep.restart)
 *                     %.9e  -- 9 digits after the point (10 significant), exact
 *                     float32 round trip, identical to the JIT literal.
 *
 *   P2 NEP_FMT_OBS    physical observables, PREDICTION AND REFERENCE ALIKE
 *                     (energy / force / virial / stress / charge / BEC /
 *                      spin-torque / dipole / polarizability, per structure or
 *                      per atom).  %.12g -- 12 significant digits: enough for
 *                      finite-difference checks of analytic derivatives and
 *                      still an exact float32 round trip.  The single most
 *                      important rule in this file: the NEP column and the
 *                      reference column of one file MUST share one format.
 *
 *   P3 NEP_FMT_STAT   training statistics (loss, RMSE) on stdout and in loss.out.
 *                     %-14.6e -- 6 significant digits in scientific notation,
 *                     one fixed field width for every column, usable over the
 *                     full dynamic range (fixed-point %f could not represent
 *                     losses below 5e-6 at all).
 *
 *   P4 NEP_FMT_CONF   configuration echo (cutoffs, ZBL radii, lambda weights,
 *                     spin_mag_scale, neighbour-count diagnostics).
 *                     %.9g -- 9 significant digits, so that the value echoed to
 *                     the log is the value actually used.
 *
 *   P5 NEP_FMT_TIME   wall-clock durations only (initialisation, training,
 *                     prediction).  %.3f -- a duration is not a physical
 *                     observable and millisecond resolution is all a human
 *                     reader needs; listing it here keeps the "no bare %f"
 *                     rule enforceable instead of leaving an exception implicit.
 *
 * HOW TO USE
 * ----------
 *   fprintf(fid, NEP_FMT_OBS_BLANK, value);      // trailing space, mid-row
 *   fprintf(fid, NEP_FMT_OBS_EOL, value);        // last column of the row
 *   printf("lambda_1 = " NEP_FMT_CONF ".\n", lambda_1);
 *
 * Do not introduce bare "%g" or "%f" in main_nep again; pick one of the four.
 *
 * NOTE ON loss.out
 * ----------------
 * Switching the statistics columns from %.5f to %.6e changed their textual
 * representation, so the file header now declares "# format_version 2".
 * The column ORDER and COUNT are unchanged; every reader that splits on
 * whitespace keeps working.
 */

#pragma once

/* ---- P1: persisted model parameters: exact float32 round trip (9 digits) ---- */
#define NEP_FMT_PARAM       "%.9e"
#define NEP_FMT_PARAM_1     "%16.9e"
#define NEP_FMT_PARAM_2     "%16.9e %16.9e"
#define NEP_FMT_PARAM_1_EOL "%16.9e\n"
#define NEP_FMT_PARAM_2_EOL "%16.9e %16.9e\n"

/* ---- P2: physical observables: prediction and reference share one format ---- */
#define NEP_FMT_OBS       "%.12g"
#define NEP_FMT_OBS_BLANK "%.12g "
#define NEP_FMT_OBS_EOL   "%.12g\n"
#define NEP_FMT_OBS_3_EOL "%.12g %.12g %.12g\n"
#define NEP_FMT_OBS_6_EOL "%.12g %.12g %.12g %.12g %.12g %.12g\n"

/* ---- P3: training statistics: one width, one notation, full dynamic range -- */
#define NEP_FMT_STAT     "%-14.6e"
#define NEP_FMT_STAT_1   NEP_FMT_STAT
/* composites insert an explicit blank so every column is exactly 15 characters wide,
   including across composite boundaries */
#define NEP_FMT_STAT_2   NEP_FMT_STAT " " NEP_FMT_STAT
#define NEP_FMT_STAT_3   NEP_FMT_STAT " " NEP_FMT_STAT " " NEP_FMT_STAT
#define NEP_FMT_STAT_6   NEP_FMT_STAT_3 " " NEP_FMT_STAT_3
#define NEP_FMT_STAT_10  NEP_FMT_STAT_6 " " NEP_FMT_STAT_3 " " NEP_FMT_STAT
#define NEP_FMT_STAT_14  NEP_FMT_STAT_10 " " NEP_FMT_STAT_3 " " NEP_FMT_STAT

/* ---- P4: configuration echo: what is logged is what is used ---------------- */
#define NEP_FMT_CONF "%.9g"

/* ---- P5: wall-clock durations (not a physical observable) ------------------ */
#define NEP_FMT_TIME "%.3f"

/* Bumped together with the loss.out statistics format (P3). */
#define NEP_LOSS_FORMAT_VERSION 2
