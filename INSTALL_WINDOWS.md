# Windows 安装

## 1. 依赖

| 组件 | 版本 | 检查 |
|---|---|---|
| Windows | 10/11 x64 | — |
| Visual Studio | 2022（含 C++ 桌面工作负载） | `vcvars64.bat` 存在 |
| NVIDIA CUDA Toolkit | 12.x（本项目在 12.8 上验证） | `nvcc --version` |
| GPU | 计算能力 ≥ 6.0（本项目 sm_89 / RTX 4060 Ti） | `nvidia-smi` |
| GNU make | 任意（本机用 `D:\wsl\make.exe`） | `make --version` |
| Python | 3.9+（只用于测试脚本） | `python --version` |

`BUILD.ps1` 会自动在 `Program Files\NVIDIA GPU Computing Toolkit\CUDA\*` 与 PATH 里找 `nvcc`，
在 `Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat` 找 MSVC，
并用 `nvidia-smi` 推断 `CUDA_ARCH`（可用 `-Arch` 覆盖）。

## 2. 装

```bat
:: 解压后
cd GPUMD-spin-V3-SOURCE-v3.0.0-r17
INSTALL.cmd                       :: 编译 + 冒烟测试
:: 或分步
BUILD.cmd -Arch sm_89 -Jobs 8
SMOKE_TEST.cmd
```

**PowerShell 提示 "running scripts is disabled on this system" 时**，用 `.cmd` 包装器
（只对本进程 Bypass，不改机器策略），或者：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\BUILD.ps1 -Arch sm_89 -Jobs 8
```

产物在 `src\`：`gpumd.exe`、`nep.exe`、`gpumd_replica.exe`。

## 3. 不需要管理员权限

整个流程只写当前目录与 `%TEMP%`；不写注册表、不装服务。

## 4. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `nvcc not found` | CUDA 未装或不在默认路径 | 装 CUDA 或把 `bin` 加进 PATH |
| `vcvars64.bat not found` | VS 版本/路径不同 | 装 VS2022 C++ 工作负载，或改 `BUILD.ps1` 里的搜索根 |
| `LNK1104: 无法打开文件 nep.exe` | 上一次的进程还占着文件 | 关掉正在跑的 `nep.exe` 再编译 |
| `C4819` 警告 | 源码含 GBK 代码页字符 | 无害，可忽略 |
| `make: command not found` | 没装 make | 装 GNU make（Chocolatey / MSYS2 / WSL 的 exe） |
| 编译通过但运行报 `cudaErrorNoKernelImageForDevice` | `-Arch` 与显卡不匹配 | 用 `-Arch sm_XX` 指定实际计算能力 |

## 5. 验证安装

```bat
SMOKE_TEST.cmd
```

期望 6 项全 PASS：单点、保守自旋、geometric、midpoint、二聚体、重启逐位、REPLICA。
本项目在全新解压目录实测：`BUILD.ps1` 218.5 s、`rc=0`、smoke 6/6 PASS。
