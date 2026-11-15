# 本机验证运行方式

这是CPU、回环HTTP和shell验证的运行约定。Windows使用PowerShell 7，调用规则遵循`powershell-safe-invocation`；从仓库根目录运行以下命令。业务/训练验收范围仍由对应模块文档决定。

## 先选用例和预算

改动对应的针对性测试先跑，失败只重跑相关部分，最后最多一轮确有必要的相关回归。先给出测试清单和执行时间预算，分别记录测试进程耗时、重复运行与环境处理；预算不足时重新规划。指南或入口的小改动只做入口烟测，前端未改时复用已通过的前端结果。GPU训练、生产服务和部署按用户授权单独安排。

## Windows CPU入口

统一用[tools/test_cpu.py](../tools/test_cpu.py)，参数原样传给unittest。`-X utf8`固定当前解释器的文本编码，入口为子进程设置`PYTHONUTF8=1`；无需修改系统区域、终端全局编码或用户环境变量。

```powershell
# 单个相关文件；按实际改动替换pattern
python -X utf8 tools/test_cpu.py discover -s tests -p test_preview_clock.py

# 只重跑相关用例
python -X utf8 tools/test_cpu.py discover -s tests -p test_hybrid_console.py -k test_shell_pipeline_retains_producer_failure

# torque200模块的相关测试
python -X utf8 tools/test_cpu.py discover -s experiments/torque200/tests -p test_workbench_validation_cache.py
```

需要整套主回归时才用`python -X utf8 tools/test_cpu.py discover -s tests`。以退出码及实际失败/跳过结果判定，不把依赖缺失的跳过写成已验证。

## Git Bash与WSL分开

Windows上的shell单测会调用Windows Python，因此使用已安装的Git Bash。入口把测试中`subprocess.run(['bash', ...])`的首个argv替换为`$env:ProgramFiles`下`Git/bin/bash.exe`的绝对路径，其他参数、断言和退出码保持原样。仅把Git目录加到PATH前面仍可能选中System32的WSL启动器，直接复用入口即可。

Git装在其他位置时，仅在当前验证进程设置`TENNY_TEST_BASH`为真实Git Bash绝对路径；已有安装可用就复用。找不到时报告该路径不可用，再定位安装，不根据一次裸`bash`失败重装工具。

直接做shell语法检查也显式指定Git Bash：

```powershell
$gitBash = Join-Path $env:ProgramFiles 'Git\bin\bash.exe'
& $gitBash '-n' 'experiments/torque200/hybrid-wsl.sh'
$checkExit = $LASTEXITCODE
if ($checkExit -ne 0) { throw "Shell syntax check failed: $checkExit" }
```

WSL仿真、训练及嵌入式构建仍使用其原入口和Linux解释器；本工具只适配Windows主机CPU测试。

## 回环HTTP权限

本机已确认：部分Codex沙箱允许启动测试服务，却禁止连接其随机回环端口，表现为`PermissionError: [WinError 10013]`。执行含真实`127.0.0.1`客户端/服务端的测试时，若本轮仍是该受限沙箱，直接使用工具允许的`require_escalated`执行方式；沿用现有授权/审核流程，权限拒绝则如实报告受阻。

例如：`python -X utf8 tools/test_cpu.py discover -s tests -p test_workbench_preview_http.py`。测试使用自己临时的回环端口及fixture，不连接生产服务。该入口处理编码与Bash选择，不提升权限。

同一受限环境的10013无需再次逐项重试、换生产地址或修改防火墙。若出现端口占用、HTTP断言失败或其他错误，按其实际错误排查，不能都归因为沙箱。

## 失败分类

| 表现 | 处理 |
| --- | --- |
| 已知沙箱回环10013 | 选择允许回环连接的执行方式 |
| Windows测试误用WSL bash | 使用仓库入口或显式Git Bash路径 |
| 子进程输出UTF-8解码失败 | 使用`-X utf8`与入口的子进程UTF-8设置；外部程序按其真实编码处理 |
| fixture缺seed/experiment_config、断言失败 | 作为测试适配或业务失败核查，不混称为环境问题 |

依据：2026-11-15本机已通过的shell退出码/UTF-8验证与回环HTTP回归；历史原始记录在忽略的`tasks/results/workspace-integration-20261115/`和`tasks/results/performance-fixes-20261115/`。这些路径只供追溯，日常验证直接使用仓库入口。
