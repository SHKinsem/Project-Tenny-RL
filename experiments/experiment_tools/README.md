# 实验通用工具

把输入核对、任务运行、配对统计和阶段策略筛选从机器人训练逻辑中分离。仅用 Python 标准库；导入时不加载 JAX、MuJoCo 或 checkpoint，不启动训练。

## 直接使用

从仓库根目录执行。省略 `--output` 时比较/筛选结果输出到终端；指定的结果文件必须不存在。

```powershell
python -X utf8 -m experiments.experiment_tools compare experiments/experiment_tools/examples/paired-study.json
python -X utf8 -m experiments.experiment_tools select experiments/experiment_tools/examples/paired-study.json --condition changed --metric rms_xy_m
python -X utf8 -m experiments.experiment_tools run experiments/experiment_tools/examples/cpu-jobs.json --dry-run
```

第三条只验证方案，不创建目录或启动进程。明确执行 CPU 示例时用一个新的输出目录：

```powershell
python -X utf8 -m experiments.experiment_tools run experiments/experiment_tools/examples/cpu-jobs.json --output tasks/results/my-cpu-example
python -X utf8 -m experiments.experiment_tools freeze experiments/experiment_tools/examples --output tasks/results/my-inputs.json
python -X utf8 -m experiments.experiment_tools check tasks/results/my-inputs.json
```

`run` 是执行器：按方案真实执行 argv。GPU、WSL 和生产任务应继续由所属适配器及生命周期控制，不能因为有通用运行器就绕过其验证。示例只写独立目录里的小型 JSON，不运行机器人。

## 两个历史报告适配器

`compare`、`normalize`、`select` 接受标准 `paired_study` 文件，也接受已支持实验的 `comparison/summary.json` 或其结果目录。目前支持 `mature_balance_finetune_summary` 与 `wheel_hold_ablation_summary` v1。

```powershell
python -X utf8 -m experiments.experiment_tools compare tasks/results/wheel-hold-ablation-20261116
python -X utf8 -m experiments.experiment_tools select tasks/results/wheel-hold-ablation-20261116 --condition weak_wheel --metric first_30s_rms_xy_displacement_m --require-full-parameters
```

这些机器上的历史结果被 Git 忽略，不是工具安装依赖。适配器只读小型 summary 和已记录的候选验证证据；不读大轨迹文件，不重算机器人物理指标。`provenance` 保留 summary 路径/SHA、原记录的训练合同与验证范围。新增机器人实现应输出相同数据合同，或新增明确格式适配器，不能猜测网络维度兼容性。

## Python 接口与数据合同

公共入口见 `__init__.py`，全部返回可 JSON 序列化的数据：

| 函数 | 输入与结果 |
|---|---|
| `freeze_inputs(paths)` / `verify_inputs(inventory)` | 显式文件/目录 SHA-256 库存及复核；发生变化时失败 |
| `validate_plan(plan)` / `run_matrix(plan, output_dir)` | 纯方案检查；顺序执行、预算、日志和任务终态 |
| `load_study(path)` / `summarize(study)` | 标准数据/历史适配；按 seed 配对汇总 |
| `select_snapshot(study, condition=..., metric=..., partition='primary', require_full_parameters=False, min_step=1)` | 在声明的分区内筛选有效阶段策略并返回证据 |

`paired_study` v1 的完整最小例子在 [examples/paired-study.json](examples/paired-study.json)。必须声明条件及 baseline/treatment、seed、互不重叠的 case 分区、指标 unit/direction，以及完整的 condition × seed × case 矩阵。每条记录包含所有指标的 `values` 和明确布尔 `eligible`；未知值用 `null`。整数 seed `0` 与字符串 `"0"` 分别处理。

统计先在同 seed 的声明初态内等权平均，再做 treatment − baseline，输出每 seed 差值、中位/范围、改善/退化/相同的数量；不会把多个固定初态当成额外训练 seed。任一 case 对某指标未知或不适用时，该 seed/分区/指标均值为未知，不丢行、不补零。固定参考策略单独放 `reference_records`，不当作一个训练 seed。结果只是描述性统计，不自动声称显著性或因果关系。

物理指标与资格由机器人适配器负责。两个历史适配器对漂移/速度/姿态要求撑满声明时域；提前倒地的小位移不能入选。累计合格时间可记录倒地前的有效累计，未测量仍为 `null`。快照选择只使用明确给定的指标/方向/分区，默认排除 step=0；默认不使用 heldout 调参。相同指标优先更早步数，再按 ID 排序。

快照的 `artifacts` 区分 `descriptor_only`、`full_parameters_verified_at_run` 和能力未知。只有历史候选确有有限值/重载一致的完整参数验证、且 descriptor SHA 匹配时，历史适配器才附完整参数标记。`--require-full-parameters` 排除其余候选；这可能只剩少数有证据的模型，不代表所有未附证据的模型都不存在。该标记是记录时的证据，远程路径/源码/执行器/完整状态仍须在真正续训前实时验证。这里没有恢复模型或优化器；历史完整候选续训仍重建 Adam。

`experiment_jobs` v1 例子见 [examples/cpu-jobs.json](examples/cpu-jobs.json)。支持总墙钟预算、逐任务超时、明确 argv、可选 cwd、`expected_result.path/checks` 和 `frozen_inputs`。只展开整项 argv `{python}` / `{run_dir}`；Python 字典花括号、引号和其他参数保持字面值。cwd 默认独立 run_dir。结果检查可使用点分字段路径，布尔与数值不同；应把结果写入本次独立目录，检查本身不会证明外部旧结果的产生时间。

输出结构：`plan.json`、`status.json`、`jobs/<id>.json`、`logs/<id>.log`、`runs/<id>/`。工具状态与训练器自己的 status 分开。失败、超时、结果不满足、冻结输入变化时停止后续任务并保存失败信息；不自动重试。超时只管理直接子进程，不负责脱离进程树的后台训练。总预算含核对开销，但不能中断磁盘操作；核对超预算时记失败，不启动下一项。

冻结库存包含目录/空目录、相对路径、大小和字节 SHA，拒绝符号链接/reparse point；它不自动发现依赖，不是原子文件系统快照，也不是现有模型兼容性源码指纹。不要冻结持续变化的输出树。

CLI 成功为退出码 0；`run` 已记录的运行失败为 1；输入/合同/文件错误为 2。JSON 拒绝重复字段和非有限值。输出文件拒绝覆盖；运行目录必须新建或为空。

## 网页接入边界与验证

本包尚未接入生产网页。后续服务可以调用这些函数并读相同 JSON；任务提交应经现有控制器，把自由 argv 留在受信任适配器内部。不要将任意命令编辑器直接暴露为网页训练功能。

功能映射和最小接入顺序见 [网页实验体验草案](../../tasks/workbench-experiment-ux-draft.md)。针对性验证入口：

```powershell
python -X utf8 tools/test_cpu.py discover -s tests -p 'test_experiment_tools_*.py'
```

测试覆盖真实 CPU 子进程、超时/失败/结果异常/输入变化、缺失时域/配对统计，以及两个小型历史格式和参数证据边界；不启动真实训练。
