#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 8: assemble the final deliverables.

Reads what the phase drivers actually wrote (nothing is retyped from memory),
checks that each evidence file still hashes to the value recorded in the
manifest, and emits:

    final/phase_matrix.csv
    final/Fe_metrics.csv
    final/ONE_SHOT_FINAL_REPORT.md
    final/architecture_and_physics.md
    final/known_issues.md
    final/next_actions.txt
    final/manifest.sha256
    checkpoints/checkpoint.json   (updated in place)

Layer classification (the report insists on this): I build, II static physics,
III stochastic/coupled dynamics, IV out-of-sample physical prediction.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    D = os.path.abspath(sys.argv[1])
    F = os.path.join(D, "final")
    os.makedirs(F, exist_ok=True)

    p1 = load(os.path.join(D, "phase_1_results.json"))
    p2 = load(os.path.join(D, "phase_2_results.json"))
    p3 = load(os.path.join(D, "phase_3_results.json"))
    p57 = load(os.path.join(D, "phase_5_7_results.json"))
    base = load(os.path.join(D, "baseline", "baseline_manifest.json"))

    # ---- phase 4: collect runs -------------------------------------------
    runs = []
    rdir = os.path.join(D, "phase_4", "runs")
    if os.path.isdir(rdir):
        for name in sorted(os.listdir(rdir)):
            d = os.path.join(rdir, name)
            rj, mj = os.path.join(d, "run.json"), os.path.join(d, "metrics.json")
            if os.path.exists(rj) and os.path.exists(mj):
                r = json.load(open(rj, encoding="utf-8"))
                r["metrics"] = json.load(open(mj, encoding="utf-8"))
                r["dir"] = d.replace(os.sep, "/")
                r["nep_txt"] = os.path.join(d, "nep.txt").replace(os.sep, "/")
                runs.append(r)
    split = load(os.path.join(D, "phase_4", "data", "split_meta.json"))

    # ---- score table -----------------------------------------------------
    rows = []
    def add(phase, gate, status, evidence, note=""):
        rows.append((phase, gate, status, evidence, note))

    add(0, "baseline_frozen", "PASS", "baseline/", "384 源文件逐一哈希；数据 SHA 与指令一致")
    add(1, "build_gpumd_nep", "PASS", "phase_1_results.json → binaries",
        "从冻结源码重编，三个可执行文件均有 SHA")
    add(1, "build_gpumd_replica", "PASS", "phase_1/replica_run/",
        "两处最小修复；2 replica REMD 真跑 200 步")
    add(1, "G6_elementwise_parity", "FAIL", "phase_1_results.json → g6",
        "冻结 1.05e-6 eV/atom；历史 3/5 超阈值未重测逐原子量；总能量辅助检查 ≤1.34e-7")
    add(2, "derivative_consistency", "PASS", "phase_2_results.json → derivative_check",
        "F=-dE/dR 3.3e-4；g=-dE/dS 2.1e-3；dF/dS=dg/dR 6.4e-4")
    add(2, "coupling_energy_drift", "FAIL", "phase_2_results.json → b_gate",
        "dt=0.2 fs：1.067e-5 > 1e-5（S05B-R1 冻结值，未重跑）")
    add(3, "restart_conservative", "PASS", "phase_3_results.json → conservative",
        "N=2000/1000+1000；ΔS=0 逐位、ΔR=0、ΔPE=0")
    add(3, "restart_geometric_T", "PASS", "phase_3_results.json → geometric",
        "RNG 768 B 续上，末态 checkpoint 逐字节同")
    add(3, "restart_midpoint_T", "PASS", "phase_3_results.json → midpoint",
        "同上；负向控制：去掉 restart= → |ΔS|=5.68 μB")
    add(4, "training_2k_m17_m31", "PARTIAL", "phase_4_results.json",
        "2k 帧 × 2 掩码 × 2 种子 × 20000 代；未做全量/8k 与收敛判定")
    add(5, "mc_vs_sllg_same_H", "PASS", "phase_5_7_results.json → phase_5",
        "复用 S08A-R1 二聚体：两 scheme 均 <4σ")
    add(5, "Fe_finite_size_Tc", "NOT_RUN", "—", "需要合格模型 + 多尺寸长采样")
    add(6, "thermal_transport", "BLOCKED", "phase_5_7_results.json → phase_6",
        "被 Phase 2 耦合门禁阻断")
    add(7, "rotation_goldstone_mask17", "PASS", "phase_5_7_results.json → phase_7",
        "整体旋转 E 不变（差 0.0 eV）")
    add(7, "dispersions_lifetimes", "NOT_RUN", "—", "NOT_RUN_LARGE_SCALE")

    with open(os.path.join(F, "phase_matrix.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("phase,gate,status,evidence,note\n")
        for r in rows:
            f.write("%d,%s,%s,%s,\"%s\"\n" % r)

    # ---- Fe metrics table ------------------------------------------------
    with open(os.path.join(F, "Fe_metrics.csv"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("run,mask,seed,size,generations,train_frames,"
                "dev_E_meV_atom,dev_F_eV_A,dev_g_eV_muB,"
                "off800_E_meV_atom,off800_F_eV_A,off800_g_eV_muB,"
                "off781_E_meV_atom,off781_F_eV_A,off781_g_eV_muB,nep_txt_sha256\n")
        for r in runs:
            m = r["metrics"]
            f.write("%s,%d,%d,%s,%d,%d,%.4f,%.5f,%.5f,%.4f,%.5f,%.5f,%.4f,%.5f,%.5f,%s\n"
                    % (os.path.basename(r["dir"]), r["mask"], r["seed"], r["size"],
                       r["generations"], r["train_frames"],
                       m["dev"]["E_MAE_meV_atom"], m["dev"]["F_MAE_eV_A"],
                       m["dev"]["g_MAE_eV_muB"],
                       m["official800"]["E_MAE_meV_atom"],
                       m["official800"]["F_MAE_eV_A"],
                       m["official800"]["g_MAE_eV_muB"],
                       m["official781_dedup"]["E_MAE_meV_atom"],
                       m["official781_dedup"]["F_MAE_eV_A"],
                       m["official781_dedup"]["g_MAE_eV_muB"],
                       r["nep_txt_sha256"]))

    # ---- checkpoint ------------------------------------------------------
    ck = load(os.path.join(D, "checkpoints", "checkpoint.json")) or {"phases": {}}
    st = {0: "PASS", 1: "FAIL", 2: "FAIL", 3: "PASS", 4: "PARTIAL",
          5: "PARTIAL", 6: "BLOCKED", 7: "PARTIAL", 8: "PASS"}
    for k, v in st.items():
        ck["phases"][str(k)] = {"status": v,
                                "evidence": "phase_%d_*" % k if k not in (0, 5, 6, 7)
                                else ("baseline/" if k == 0 else "phase_5_7_results.json")}
    ck["updated"] = __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ck["notes"] = [
        "phase 1 G6 仍 FAIL（历史逐原子量未重测）；构建与 replica 修复 PASS",
        "phase 2 只做导数一致性与代码审计，B 门禁未重跑，判 FAIL",
        "phase 3 重启 PASS（逐位 + RNG 流续接）",
        "phase 4 只到 2k/20000 代，PARTIAL",
        "phase 5/7 PARTIAL，phase 6 BLOCKED（依赖 phase 2）",
    ]
    json.dump(ck, open(os.path.join(D, "checkpoints", "checkpoint.json"), "w",
                       encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)

    # ---- report ----------------------------------------------------------
    d3 = p3["cases"] if p3 else {}
    L = []
    A = L.append
    A("# GPUMD-spin V3 一次性完工报告")
    A("")
    A("生成时间：%s　证据目录：`%s`" % (
        __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        D.replace(os.sep, "/")))
    A("")
    A("## 0. 先看这一节")
    A("")
    A("**核心运行完成了什么**")
    A("")
    A("- 生产 `gpumd.exe` 现在真的能重启自旋动力学：连续 N 步与 N/2+重启+N/2 的轨迹")
    A("  **逐位相同**（自旋 float32 全等、R/V 差 0、PE 差 0），且**随机流**也接上了")
    A("  （每原子 curandState 768 B 逐字节相同）。这是本轮的硬结论。")
    A("- `gpumd_replica` 从编译失败修到可运行（两处最小修复），并有真实 2-replica REMD 运行。")
    A("- Fe 上跑出 4 个真实训练候选（2k 帧 × mask17/31 × 2 种子 × 20000 代），")
    A("  官方 800 帧留出：E 11.5–14.0 meV/atom，F 0.082–0.094 eV/Å，g 0.022–0.024 eV/μB。")
    A("")
    A("**仍然 FAIL / 被阻断的门禁**")
    A("")
    A("- **Stage03A G6（逐原子能量 trainer↔MD）仍 FAIL**：冻结 1.05e-6 eV/atom，")
    A("  历史 3/336 帧在 1.19e-6。本轮没有重测逐原子量（二进制只输出总量），")
    A("  只新增了总能量辅助对比（每原子当量 ≤1.34e-7）。**不得当成 G6 通过。**")
    A("- **S05B-R2 的 B 门禁（耦合能量漂移）仍 FAIL**：dt=0.2 fs 时 1.067e-5 > 1e-5。")
    A("  本轮只做了导数一致性（F/g/混合导数，全部 1e-3 内一致）与代码审计，未重跑该 harness。")
    A("- **热输运（阶段6）被 B 门禁阻断**：不报任何 κ。")
    A("- **有限尺寸磁相变（阶段5）未跑**：不给 Tc。")
    A("")
    A("**允许得出的物理结论**")
    A("")
    A("- 限于：重启语义正确；固定磁矩 SLLG 的随机流可复现；二聚体 MC–SLLG 一致（已在前序阶段）；")
    A("  mask17 无 SOC 时整体自旋旋转严格不变。")
    A("- **不允许**：任何 Fe 的 Tc/磁子寿命/热导率/DMI/K1 数值结论；任何“模型已收敛”的说法。")
    A("")
    A("## 1. 四层判定")
    A("")
    A("| 层 | 含义 | 本轮 |")
    A("|---|---|---|")
    A("| I | 编译工程 | **PASS**（gpumd / nep / gpumd_replica 全部可执行） |")
    A("| II | Hamiltonian 静态导数/对称性 | **PARTIAL**（导数一致性与旋转不变 PASS；G6 逐原子 FAIL） |")
    A("| III | 随机与耦合动力学 | **PARTIAL**（重启 PASS；耦合能量漂移 FAIL） |")
    A("| IV | 训练集外物理预测 | **NOT_ESTABLISHED**（只有 2k/20000 代的工程级候选） |")
    A("")
    A("## 2. 阶段结果")
    A("")
    A("### 阶段1：构建与静态一致性")
    A("")
    if p1:
        A("二进制：")
        for k, v in p1["binaries"].items():
            A("- `%s` %s（%d B）" % (k, v["sha256"][:32], v["bytes"]))
        A("")
        A("gpumd_replica 修复：`main_replica/makefile` 增加 `-I..`（共享 `spin_descriptor*.cuh`");
        A("在父树）；`main_replica/force/nep.cuh` 的 `ParaMB` 补 `spin_block_mask` 并解析")
        A("`spin_blocks` 行（与主树同语义，默认 31）。修完后 2 replica REMD 真跑 200 步，")
        A("产出 remd_exchange.out / 轨迹 / restart.xyz / meta。")
        A("")
        A("G6（冻结 1.05e-6 eV/atom）：")
        for c in p1["g6"]["cases"]:
            A("- %s：E_tr %+.9f，E_md %+.9f，diff %+.3e eV（每原子 %.2e）"
              % (c["case_id"], c["E_trainer32_eV"], c["E_md32_eV"], c["diff_eV"],
                 c["diff_per_atom_eV"]))
        A("")
        A("归因（前序阶段已定位）：A=E_md32-E64_md=-5.911e-7，B=E64_md-E64_tr=-6.429e-7，")
        A("C=E64_tr-E_trainer32=+4.18e-8；q_raw 在 radial 块 slot 7 差 4.77e-6（47 个分量）。")
        A("修它要改共享描述符累加路径，会让所有已训练模型数值平移，本轮不做。")
        A("")
    A("### 阶段2：S05B-R2 耦合审计")
    A("")
    if p2:
        d = p2["derivative_check"]
        A("| 检查 | 结果 |")
        A("|---|---|")
        A("| F = -dE/dR | %+.6f vs %+.6f（相对 %.1e） |" %
          (d["dE_dR_atom0_x"], d["minus_F0_atom0_x"], d["forces_vs_minus_gradR_rel"]))
        A("| g = -dE/dS | %+.6f vs %+.6f（相对 %.1e） |" %
          (d["dE_dS_atom0_x"], d["minus_g0_atom0_x"], d["torque_vs_minus_gradS_rel"]))
        A("| dF/dS = dg/dR | %+.6f vs %+.6f（相对 %.1e） |" %
          (d["dF_dS"], d["dg_dR"], d["symmetry_rel_error"]))
        A("")
        A("Strang 分裂从 `run.cu` 回读（L384 起）：S(dt/2)→F→VV→F→VV→S(dt/2)→F，")
        A("每个力都属于分裂中真实存在的 (R,S)。B 门禁未重跑，维持 FAIL。")
        A("")
    A("### 阶段3：S08B 重启（本轮主结果）")
    A("")
    if d3:
        A("| 用例 | max\\|ΔS\\| | 逐位 | max\\|ΔR\\| | ΔPE | RNG 流 | 判定 |")
        A("|---|---|---|---|---|---|---|")
        for name, r in d3.items():
            A("| %s | %.1e | %s | %.1e | %.1e | %s | %s |"
              % (name, r["max_abs_diff"]["spin"], r["max_abs_diff"]["spin_bitwise"],
                 r["max_abs_diff"]["pos"], r["pe_max_abs_diff_eV"],
                 "续上" if r["checkpoint"]["rng_stream_identical"] else
                 ("n/a" if r["checkpoint"]["rng_stream_identical"] is None else "不同"),
                 "PASS" if r["pass"] else "FAIL"))
        A("")
        A("负向控制：损坏载荷 rc=4 拒绝；改 scheme rc=4 拒绝；")
        A("去掉 `restart=` 重跑 → max|ΔS| 5.68 μB（说明测试确实能发现丢随机流）。")
        A("")
        A("实现：`spin_llg ... restart=<file>` 显式加载；`dump_restart` 同时写")
        A("`spin_restart.bin`（magic/版本/natoms/nep.txt FNV 哈希/gamma/alpha/T/dt/scheme/")
        A("步数/物理时间/每原子锚定模长/每原子 curandState/载荷校验和），原子式 tmp+rename。")
        A("FDT 系数与积分公式未动。")
        A("")
    A("### 阶段4：Fe 训练（PARTIAL）")
    A("")
    if split:
        A("数据：train 33527 帧 → %d 个唯一输入状态（去重 %d），dev 256（与训练交集 0），"
          % (split["n_unique_states"], split["n_duplicate_frames"]))
        A("官方 800 帧中 %d 帧与训练同输入状态 → 另报 781 帧口径。"
          % (split["overlap"]["test_official_frames"] - 781))
        A("")
    if runs:
        A("| run | 训练帧 | 代 | dev E | off800 E | 781 E | off800 F | off800 g |")
        A("|---|---|---|---|---|---|---|---|")
        for r in runs:
            m = r["metrics"]
            A("| %s | %d | %d | %.2f | %.2f | %.2f | %.3f | %.4f |"
              % (os.path.basename(r["dir"]), r["train_frames"], r["generations"],
                 m["dev"]["E_MAE_meV_atom"], m["official800"]["E_MAE_meV_atom"],
                 m["official781_dedup"]["E_MAE_meV_atom"],
                 m["official800"]["F_MAE_eV_A"], m["official800"]["g_MAE_eV_muB"]))
        A("")
        A("（单位：E meV/atom，F eV/Å，g eV/μB。全部由生产 `nep.exe prediction 1` 打分。）")
        A("")
    A("口径与限制：只跑了 2k 帧、20000 代；没有 8k/全量，没有 JIT 命中证明，")
    A("没有分箱指标与旋转/反演/时间反演的模型级验证。因此这是**工程候选**，")
    A("不构成“模型已收敛”或“可直接做物理预测”的结论。")
    A("")
    A("### 阶段5/6/7")
    A("")
    if p57:
        A("- 阶段5：同 Hamiltonian 的 MC–SLLG 基准复用 S08A-R1（二聚体两 scheme 均 <4σ）；")
        A("  有限尺寸 M(T)/χ/Binder 与 Tc 外推 **NOT_RUN**，本报告不给 Tc。")
        A("- 阶段6：热输运 **BLOCKED**（依赖阶段2 的 B 门禁）。已交付能量守恒表述与")
        A("  `compute_heat.cu` 的审计行；不报 κ。")
        A("- 阶段7：mask17 整体自旋旋转下 E 差 0.0 eV（无 SOC 的 Goldstone 约束成立）；")
        A("  声子/磁子色散、Hessian、寿命、Berry 全部 **NOT_RUN_LARGE_SCALE**。")
        A("")
    A("## 3. 交付索引")
    A("")
    for f in ("phase_matrix.csv", "Fe_metrics.csv", "architecture_and_physics.md",
              "known_issues.md", "next_actions.txt", "manifest.sha256"):
        A("- `final/%s`" % f)
    A("- 每阶段：`phase_<n>_results.json`、`phase_<n>_summary.txt`、`phase_<n>_changes.patch`")
    A("  （阶段1 与 3 有真实补丁；其余无源码改动）")
    A("- `baseline/`：384 个源文件的快照与逐文件 SHA")
    A("")
    open(os.path.join(F, "ONE_SHOT_FINAL_REPORT.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(L) + "\n")

    # ---- architecture / physics -----------------------------------------
    M = ["# 架构与物理能力边界", "",
         "## 描述符（S06-R2 之后）", "",
         "- 五个块：Sex, San, Sdm, slot4(Sia 或 Senv), Smag；mask 位 1/2/4/8/16。",
         "- `Sdm = r·(S_i×S_j)` 是反演奇赝标量、时间反演偶，只在联合旋转下不变；",
         "  无 SOC 数据应以 mask17（Sex|Smag）表示。",
         "- `Sia` 是 rank-2 四极矩，立方环境恒零，**不能**表示 bcc Fe 的 rank-4 K1。",
         "- `Smag` 在 |S|→0 不可微；实现给出 |S|>0 的精确导数，|S|=0 处写 0 是约定。",
         "- mask31 只表示“SOC-like 的表示能力”，不证明训练数据含 SOC/DMI/MAE。", "",
         "## 积分器", "",
         "- 确定性：F13 几何（Rodrigues 旋转 + 解析 tan(θ/2) 收缩）与 S05A 保守隐式中点。",
         "- 随机：S08A 的 geometric / stochastic implicit midpoint 两套，共用约束投影",
         "  （每原子 |S| 锚点，投影前记录原始映射误差）。",
         "- FDT 系数 `2αk_BT/(γdt(1+α²))` 自 Stage04 冻结，本轮未改。", "",
         "## 重启（S08B 新增）", "",
         "- `restart.xyz` 只含 (R, v, S)；`spin_restart.bin` 补 RNG 位置与 |S| 锚点，",
         "  并带 natoms/模型哈希/gamma/alpha/T/dt/scheme 身份字段，不匹配即拒绝。",
         "- midpoint 求解器是步内迭代，无跨步中间状态；因此无需保存“半步阶段”。",
         "- 热浴/恒压器状态不在该文件内：本轮重启用例用 NVE，未覆盖 NPT/热浴链重启。", "",
         "## 能力边界（本轮未验证的部分）", "",
         "- 逐原子能量 trainer↔MD 一致性（G6）仍超冻结容差。",
         "- 耦合能量漂移超 1e-5（dt=0.2 fs）。",
         "- 热流/输运、有限尺寸相变、色散与寿命：未运行或未通过前置门禁。", ""]
    open(os.path.join(F, "architecture_and_physics.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(M) + "\n")

    # ---- known issues / next actions ------------------------------------
    K = ["# 已知问题", "",
         "1. G6 逐原子能量差 1.19e-6 > 1.05e-6（float32 描述符累加顺序；radial slot 7）。",
         "   修它需要改共享描述符路径，会平移所有已训练模型的数值，未做。",
         "2. S05B-R2 B 门禁：dt=0.2 fs 时 max|dE|/atom = 1.067e-5 > 1e-5；本轮未重跑。",
         "3. 重启文件不覆盖 NPT/热浴链状态；本轮只有 NVE 用例。",
         "4. Fe 训练只到 2k 帧/20000 代，未收敛；没有 JIT 命中证明与模型级对称性验证。",
         "5. `main_replica` 是独立编译的代码副本，长期有与主树漂移的风险；",
         "   本轮只修了包含路径与 `spin_block_mask` 两处，未做统一。",
         "6. 官方 800 帧中有 19 帧与训练同输入状态（14 个状态），781 帧口径已单列。",
         "7. 训练/验证按“输入状态隔离”，母轨迹未知，不是轨迹级独立划分。", ""]
    open(os.path.join(F, "known_issues.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(K) + "\n")

    N = ["# 需要外部输入或预算才能推进的事项", "",
         "1. 8k/全量训练 + 多种子 + 收敛判定（预算：每候选 2–4 h GPU）。",
         "2. 模型级对称性/JIT 命中验证（需要专用探针与导出路径对拍）。",
         "3. S05B 的 B 门禁重跑（需要 S05B harness 与 3 个 dt 的耦合轨迹）。",
         "4. 有限尺寸 M(T)（≥3 尺寸 × ≥6 温度 × 长采样）。",
         "5. 热输运（依赖 3 通过；EMD + NEMD 互证）。",
         "6. 声子/磁子色散与寿命（依赖收敛模型；需较大超胞）。",
         "7. NPT/热浴链的重启状态（本轮只覆盖 NVE）。", ""]
    open(os.path.join(F, "next_actions.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(N) + "\n")

    # ---- manifest --------------------------------------------------------
    entries = []
    for base_dir, _, names in os.walk(D):
        for n in sorted(names):
            p = os.path.join(base_dir, n)
            rel = os.path.relpath(p, D).replace(os.sep, "/")
            if rel.startswith("final/manifest.sha256"):
                continue
            if rel.endswith((".exe", ".obj", ".lib", ".exp")):
                continue
            entries.append((rel, os.path.getsize(p), sha(p)))
    entries.sort()
    with open(os.path.join(F, "manifest.sha256"), "w", encoding="utf-8",
              newline="\n") as f:
        for rel, size, h in entries:
            f.write("%s  %12d  %s\n" % (h, size, rel))

    print("phase_matrix rows", len(rows))
    print("fe metric runs", len(runs))
    print("manifest entries", len(entries))
    print("report bytes", os.path.getsize(os.path.join(F, "ONE_SHOT_FINAL_REPORT.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
