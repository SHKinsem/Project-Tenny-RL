# Workbench integration

- 执行CPU、回环HTTP或shell验证前，先读[验证指南](docs/validation.md)，复用其中的Windows测试入口、UTF-8、Git Bash及权限规则；先选相关用例和时间预算，再运行。

- 开始整合/部署前读PROGRESS.md；行为决定见tasks/workbench-production-draft.md，按阶段更新tasks/todo.md。
- 用户已授权生产接入。以WSL实际已部署字节为基准，保留原树未部署实验改动；核对进程与配置后再切换服务，保护活跃训练。
- 模型兼容依赖源码指纹。整合接点在网页分发、读取适配、timed包装层及新模块；受保护源码/资产的清单与基线哈希在忽略的tasks/results/workbench-production/中。
- 512新训练默认、旧checkpoint五项配置继承、原生评估、阶段计时及缓存按原合同保留。修改这些行为前读对应RUNTIME-DEFAULTS.md专题，而非推断。
- 新UI来自真实API，未实现控制禁用。机器人/任务/动作语义来自catalog/descriptor，兼容性由真实验证器决定，不按数组维度猜测。
- 生产变更先精确备份并准备回退；Git main合并/推送按CLAUDE.md处理。公开文档使用可移植路径，机器状态放忽略的结果目录。
