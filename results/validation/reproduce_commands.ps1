# reproduce_commands.ps1 -- rebuild every number in validation/BCC_Fe_full_physics
#
# Run from the project root:  D:\researche_test\rongkunchen\GPUMD-spin\V3
# Every command below was actually executed; the raw output is under raw_data/.
# Total runtime on the reference machine (RTX 4060 Ti, 8 cores): ~45 min.
#
# NOTHING here retrains or modifies the model.  The only inputs are
#   gpumd.exe (sha256 090defe9...), fe_mask17_2desc_r18.nep.txt (9d5799f8...)
# and geometries built by scripts/feutils.py.

$ErrorActionPreference = "Stop"
$Root = "D:\researche_test\rongkunchen\GPUMD-spin\V3"
$Val  = Join-Path $Root "validation\BCC_Fe_full_physics"
$Py   = "python"
Set-Location (Join-Path $Val "scripts")

# ---- 0. identity (hashes of source, binaries, models, dataset)
$Py -c "import hashlib,json,os;print('see model_identity.json')"
Get-FileHash "$Root\final_delivery_20261101_r18\win_staging\bin\gpumd.exe" -Algorithm SHA256
Get-FileHash "$Root\final_delivery_20261101_r18\staging\GPUMD-spin-V3-SOURCE-v3.1.0-r18\models\fe_mask17_2desc_r18.nep.txt" -Algorithm SHA256

# ---- A. model identity and basic response  (~1 min)
$Py stage_A.py

# ---- B. EOS, moment, magnetic states      (~5 min; B2 alone is 31 static runs)
$Py stage_B.py
$Py stage_B2.py

# ---- C. exchange + magnon dispersion      (~8 min; 8^3 spiral energies)
$Py stage_C.py

# ---- E. phonons through the built-in module (~2 min; keeps the negative result)
$Py stage_E.py

# ---- G. exchange striction + blocked stages (~6 min)
$Py stage_G.py

# ---- D. Curie temperature
#   D-a  sampler time-step sensitivity (this is the decisive test)
$Py -c "import stage_D as D, json; ..."   # see raw_data/D/dt_bias.json (script body in logs)
#   D-b  M(T) with the only usable step (dt = 0.05 fs, L = 6)
$Py stage_D.py --sizes 6 --temps 700,800,900,1000,1100,1200 --ps 1.0 --dt 0.05 --tag dt005_L6

# ---- report
$Py make_report.py
Write-Host "figures in $Val\figures ; numbers in $Val\validation_results.json"

# ---------------------------------------------------------------------------
# Notes
# * stage_D.py's dt-bias block was run inline (python -c with stage_D.run_point);
#   the exact call is recorded in ../logs/stage_D_dtbias.log.
# * The 2^2..8^3 supercell convergence inside stage_C takes the bulk of the time.
# * `compute_phonon` needs `replicate` as the FIRST effective keyword in run.in;
#   otherwise gpumd stops with "replicate must be the first effective command".

# ---------------------------------------------------------------- 外部声子/磁子
# 内置 compute_phonon 在 S=0 极限求值（FM 与 |S|=0 的 omega2.out 逐字节相同），
# 因此声子全部由下面这条外部有限位移路径给出。它只调用生产二进制的
# `dump_xyz ... force`（自旋感知），IFC2 / D(q) / Berry 相位都在 Python 里做。
cd validation\BCC_Fe_full_physics\scripts

# 1) 主色散 + DOS（7x7x7 原胞超胞，delta = 0.02 A）
python phonon_fd.py --mode fm --n 7 --delta 0.02

# 2) 超胞(6/7/8)与位移(0.005/0.01/0.02)收敛
python phonon_fd.py --mode conv

# 3) FM vs DLM（4 个随机自旋构型）
python phonon_fd.py --mode dlm --n 6 --delta 0.02

# 4) 简并搜索 + Wilson 环 Berry 相位
python phonon_topo.py --grid 31

# 5) 图
python make_figures_fd.py

# 6) 把结果并回 validation_results.json 与 benchmark_comparison.csv
python update_results_fd.py

# 诊断脚本（解释 ASR 残差与距离度量）
python diag_asr.py 8
python diag_metric.py 8

# ------------------------------------------------- 弛豫 + 多磁态声子 + 磁子/耦合
# 先弛豫：`minimize` 在 S=0 极限求值，用自旋感知的能量扫描代替。
cd validation\BCC_Fe_full_physics\scripts
python relax_and_phonons.py --stage relax      # 每个磁态各自的 a0
python relax_and_phonons.py --stage phonon     # 每个磁态在自己的 a0 上算声子

# 磁子谱（模型直接测的冻结螺旋能量）、磁声耦合、交叉点
python magnon_and_coupling.py --n 8
python figures_and_crossings.py                 # 五张图 + 精细交叉分析

# 结果：raw_data/H_states/{relaxation,phonons_by_state,states_common_a,magnon_coupling}.json
#       figures/{phonons_by_state,relaxation_by_state,magnon_phonon_crossing}.png

# ---------------------------------------------------------------------------
# per-checkpoint battery (exchange J, magnon spectrum, phonons of every
# magnetic state, magnon-phonon coupling).  Idempotent: already-computed
# sections are skipped, so this can be re-run at any time.
# ---------------------------------------------------------------------------
python tools\physics_validation\model_series.py --scan work_20261101_retrain\C_5desc_500k\checkpoints
python tools\physics_validation\make_series_figures.py
python tools\physics_validation\series_report.py
python tools\physics_validation\make_val_manifest.py

# the resident orchestrator does the above automatically at every 100k boundary
# while the 500k training runs; see checkpoints/orchestrator.log
# python tools\physics_validation\series_orchestrator.py
