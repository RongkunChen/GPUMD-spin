# 从源码构建

## 1. 三个目标

| 目标 | 产物 | 入口 |
|---|---|---|
| `gpumd` | `src/gpumd.exe` | `main_gpumd/main.cu` |
| `nep` | `src/nep.exe` | `main_nep/main.cu` |
| `gpumd_replica` | `src/gpumd_replica.exe` | `main_replica/main.cu` |

三者共用 `force/`、`model/`、`integrate/`、`measure/`、`spin/`、`utilities/`。
`main_replica/` 是**独立副本**（自己的 makefile 与部分头文件），改共享头时要单独验它的构建。

## 2. Windows（已验证）

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd src
make OS=Windows_NT CUDA_ARCH=-arch=sm_89 -j 8 gpumd
make OS=Windows_NT CUDA_ARCH=-arch=sm_89 -j 8 nep
make OS=Windows_NT CUDA_ARCH=-arch=sm_89 gpumd_replica
```

`BUILD.ps1` 就是把这三条包起来，附带依赖发现、日志与退出码。

## 3. Linux（未验证）

```bash
cd src
make CUDA_ARCH="-arch=sm_80" -j8 gpumd nep
make CUDA_ARCH="-arch=sm_80" gpumd_replica
```

**本机没有验证过 Linux 构建**，`build.sh` / `test.sh` 只作参考，状态 `UNTESTED`。

## 4. 增量构建

`make` 只看 `.cu`/`.cuh` 时间戳；改头文件后如果行为没变，先 `make clean`（见 `DEVELOPER_GUIDE.md` 的坑）。

## 5. 产物自检

```bash
sha256sum src/gpumd.exe src/nep.exe src/gpumd_replica.exe
python tests/smoke.py --bin src --json smoke_result.json
```
