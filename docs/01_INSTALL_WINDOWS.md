# 01 — Windows 安装（实测路径：Windows 11 + VS2022 + CUDA 12.8）

本文的每条命令都在交付机（RTX 4060 Ti，sm_89，CUDA 12.8，MSVC 19.4x）上真实执行过。
安装过程**不需要管理员权限**，全部落在用户目录。

## 1. 依赖

| 依赖 | 版本 | 检查命令 | 说明 |
|---|---|---|---|
| Windows | 10/11 x64 | `winver` | — |
| NVIDIA 驱动 | ≥ CUDA 12.x 对应版本 | `nvidia-smi` | 需支持目标 `sm_XY` |
| CUDA Toolkit | 12.8（实测） | `nvcc --version` | 提供 `nvcc`、`cublas`、`cusolver`、`cufft` |
| MSVC | VS2022（含 "Desktop development with C++"） | `vswhere -latest -property installationPath` | 提供 `cl.exe` 与 Windows SDK |
| GNU make | 4.x（实测 4.2） | `make --version` | `src/makefile` 是 GNU make 语法；`nmake` 不通用 |
| Python | 3.9+（实测 3.13） | `python --version` | 仅测试/分析脚本需要 |

> 如果机器上只有 `nmake`：`src/CMakeLists.txt` 也保留，但本轮**未验证** CMake 路径，标 `UNTESTED`。

## 2. 自动安装（推荐）

```powershell
# 解压后进入源码包根目录
cd <解压目录>\GPUMD-spin-V3-SOURCE-v3.0.0-r17
.\INSTALL.ps1                 # 检查依赖 -> 编译 -> smoke 测试
```

**如果 PowerShell 报 "running scripts is disabled on this system"**（默认策略常是 Restricted；
本机实测就是这样），用同目录下的 cmd 包装器 —— 它只对**当前进程**加 `-ExecutionPolicy Bypass`，
不改机器策略、不需要管理员权限：

```bat
INSTALL.cmd
:: 或只编译
BUILD.cmd -Arch sm_89
:: 或只测
SMOKE_TEST.cmd
```

`INSTALL.ps1` 依次做三件事：

1. 打印 `nvcc / make / nvidia-smi / python` 的解析结果；
2. 调 `BUILD.ps1`：自动找 `vcvars64.bat`、自动按 `nvidia-smi --query-gpu=compute_cap` 选 `-arch=sm_XY`，编译 `gpumd`、`nep`、`gpumd_replica`；
3. 跑 `tests/smoke.py`（6 项），任一失败返回非零。

成功的标志：

```
== BUILD OK ==
  gpumd.exe            <sha256>  <bytes>
  nep.exe              <sha256>  <bytes>
  gpumd_replica.exe    <sha256>  <bytes>
T1_single_point    PASS ...
...
SMOKE PASS
== INSTALL OK ==
```

## 3. 手工安装

```powershell
# 1) 载入 MSVC 环境（路径按本机 VS 版本调整）
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# 2) 进入 src 并构建（OS=Windows_NT 决定 .obj 后缀与 cl 主机编译器）
cd src
make OS=Windows_NT CUDA_ARCH=-arch=sm_89 -j 8 gpumd
make OS=Windows_NT CUDA_ARCH=-arch=sm_89 -j 8 nep
make OS=Windows_NT CUDA_ARCH=-arch=sm_89    gpumd_replica

# 3) 冒烟
cd ..
python tests\smoke.py --bin src --json smoke_result.json
```

`CUDA_ARCH` 必须与 GPU 匹配：RTX 30/40 系列 `sm_86`/`sm_89`，A100 `sm_80`，H100 `sm_90`。
用错架构不会报错，但运行期会 `no kernel image is available for execution on the device`。

## 4. 关键环境变量

| 变量 | 作用 | 示例 |
|---|---|---|
| `CUDA_PATH` | `BUILD.ps1` 找 `nvcc` 的第二顺位 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` |
| `PATH` | 需含 `nvcc.exe` 与 `make.exe` | 由 `vcvars64.bat` + 你的 make 目录提供 |
| `SPIN_NOISE_DEBUG` | 设 1 时把每步热噪声写到 `spin_noise.out`（诊断 FDT 用） | `$env:SPIN_NOISE_DEBUG="1"` |

## 5. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `nvcc fatal : Cannot find compiler 'cl.exe'` | 没有先 call `vcvars64.bat` | 用 `BUILD.ps1`，或手工先 call |
| `fatal error C1083: 无法打开包括文件 "utilities/spin_descriptor.cuh"` | 在 `src/main_replica` 里单独 make，缺 `-I..` | 用顶层 `make ... gpumd_replica`（v3.0.0 已修） |
| `nvcc fatal: Unsupported gpu architecture 'sm_89'` | CUDA < 11.8 | 升级 CUDA，或改用匹配的 `sm_XY` |
| `error: identifier "curandState" is undefined` | 头文件路径被裁剪 | 用本包 `src/`，不要只拷部分文件 |
| `make: *** No rule to make target 'OS=Windows_NT'` | 用了 `nmake` | 换 GNU make |
| 运行期 `CUDA driver version is insufficient` | 驱动比 Toolkit 老 | 升级驱动 |
| 运行期 `out of memory` | 体系过大 | 减小 `batch`/原子数，或换卡 |

## 6. 无管理员权限的替代

- CUDA Toolkit 可选 “自定义安装 → 只勾 Runtime + Development”，装到用户目录并在 `PATH` 里前置。
- MSVC 可用 Build Tools（不需要完整 VS IDE），`vswhere` 同样能定位。
- 本包不写入注册表、不写系统目录；卸载 = 删除解压目录。

## 7. 本机实测记录

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti，8 GB，CC 8.9 |
| CUDA | 12.8（`nvcc --version` 见 `build.log`） |
| 构建命令 | 见 `BUILD.ps1` 生成的 `build.log` |
| 构建耗时 | 全量约 6–9 min（`-j 6`，含 `main_replica`） |
| fresh-extract 验收 | 见 `docs/13_VALIDATION_AND_KNOWN_ISSUES.md` 与 `FINAL_REPORT.md` |
