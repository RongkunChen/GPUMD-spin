# 依赖说明

## 构建与运行（必需）

| 组件                | 版本                                      | 用途                                         |
| ----------------- | --------------------------------------- | ------------------------------------------ |
| CUDA Toolkit      | >= 11.0（实测 12.2 Windows / 12.3 WSL2）    | `nvcc`、`cublas`、`cusolver`、`cufft`         |
| NVIDIA GPU        | 计算能力 >= 6.0（实测 sm_89 / RTX 4090 Laptop） | 执行                                         |
| C++ host compiler | MSVC (VS2022) 或 g++ (>= C++14)          | nvcc 的 host compiler                       |
| GNU make          | 任意近期版本                                  | `src/makefile`、`src/main_replica/makefile` |
| Windows SDK 10    | 任一版本（仅 Windows）                         | `corecrt.h` / `ucrt.lib`                   |

可选：CMake >= 3.24（只构建 `gpumd` / `nep` / `gnep`，不含 `gpumd_replica`）。

不需要：MPI、OpenMP、Python（运行时）。

## 后处理（可选）

见 `postprocessing/requirements.txt`：Python >= 3.7 + matplotlib >= 3.0。

## 计算能力对照

| GPU           | `-arch=` |
| ------------- | -------- |
| V100          | sm_70    |
| RTX 20xx / T4 | sm_75    |
| A100          | sm_80    |
| RTX 30xx      | sm_86    |
| RTX 40xx      | sm_89    |
| H100          | sm_90    |

低于 6.0 需在 `src/makefile` 的 `CFLAGS` 追加 `-DUSE_KEPLER`。

## 许可证

`LICENSES/`（上游原样保留）。GPUMD 为 LGPL v3 或更高版本。
