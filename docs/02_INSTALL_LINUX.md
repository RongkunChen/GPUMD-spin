# 02 — Linux 安装（UNTESTED）

> **状态：UNTESTED。** 本文件给出上游 GPUMD 的标准 Linux 构建方式与本包新增的构建脚本，
> 但**没有在本机 Linux 上执行过**。不要把它写成 `LINUX PASS`。
> 交付机的实测环境是 Windows 11 + MSVC 2022 + CUDA 12.8（见 `01`）。

## 1. 依赖

| 依赖 | 建议版本 | 检查 |
|---|---|---|
| Ubuntu | 22.04 / 24.04 | `lsb_release -a` |
| NVIDIA 驱动 | ≥ 535 | `nvidia-smi` |
| CUDA Toolkit | 12.x | `nvcc --version` |
| g++ | ≥ 9（支持 C++14） | `g++ --version` |
| GNU make | ≥ 4.2 | `make --version` |
| Python | 3.9+ | `python3 --version` |

```bash
sudo apt update
sudo apt install -y build-essential make python3 python3-numpy
# CUDA 按 NVIDIA 官方 runfile 或 apt 仓库安装；注意不要让 apt 覆盖驱动
```

## 2. 构建

```bash
cd GPUMD-spin-V3-SOURCE-v3.0.0-r17
CUDA_ARCH=-arch=sm_86 JOBS=$(nproc) bash build.sh     # 或在 src/ 里手工 make
```

等价手工命令：

```bash
cd src
make CUDA_ARCH=-arch=sm_86 -j"$(nproc)" gpumd
make CUDA_ARCH=-arch=sm_86 -j"$(nproc)" nep
make CUDA_ARCH=-arch=sm_86 gpumd_replica
```

- 与 Windows 的差别：不带 `OS=Windows_NT`（对象后缀 `.o`，g++ 主机编译器）；
  链接库为 `-lcublas -lcusolver -lcufft`（`main_replica` 只需 `-lcufft`）。
- `gpumd_replica` 的子 makefile 需要 `-I..`（v3.0.0 已加）；若你手工改过它，注意别把这一行删掉。

## 3. 测试

```bash
python3 tests/smoke.py --bin src --json smoke_result.json
```

## 4. 已知的 Linux 特有风险（未验证，供排查）

| 风险 | 说明 | 处理 |
|---|---|---|
| `main_replica` 的 `-lcufft` 与主树不同 | 上游设计如此 | 保留 |
| 大小写敏感 | `makefile` 是小写；`Makefile` 不存在 | 用 `make -f makefile` 或 `build.sh` |
| `make` 并行下 `.o` 竞态 | 上游 makefile 无 order-only 依赖 | 若失败，降 `-j` |
| `sm_XY` 不匹配 | 运行期 `no kernel image` | 按 `nvidia-smi` 的 CC 设 `CUDA_ARCH` |

## 5. 与 Windows 的一致性

两端使用同一份 `src/`、同一份 `nep.txt`、同一份 `run.in`。
**逐位一致只在同构建同架构内成立**；跨平台只能要求容差一致（浮点求和顺序不同）。

## 6. 状态

| 项 | 状态 |
|---|---|
| 脚本存在 | 是（`build.sh`/`test.sh`） |
| 本机执行 | **未执行** |
| 结论 | `UNTESTED` |
