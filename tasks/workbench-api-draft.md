# 工作台后端接口草稿

日期：2026-11-15。进入后端准备与首批接口补齐阶段；不部署、不接UI生产数据、不操作真实训练。

## A-01：先核对当前实现

- 已有网页提供GET /state、/learning-data、/stream.mjpg和POST /training、/control；original/torque与hybrid/teacher有两套handler。
- /training当前只分start/stop；/control的pause/unpause控制预览，不是训练。控制器已有runs/latest查询和准备/验证基础，但网页未形成新工作台合同。
- TrainingController构造会创建console目录；status/runs/latest会持锁并可能修复active.json。新的独立只读接口不实例化或调用控制器，不触发这些副作用。
- 当前后端核心文件存在并行修改，首批新增独立模块和测试，不覆盖现有handler、runner或管理器。

## A-02：首批只读接口

- 新模块位于experiments/torque200/workbench_api/，只用标准库和轻量模式定义，无ML/仿真/训练导入。
- 首批版本化GET：/api/workbench/v1/state、/checkpoints、/metrics、/capabilities。只读已有console产物，测试全部使用临时合成文件；独立HTTP测试只绑定自己的随机回环端口。
- 接口明确schema版本、来源、更新时间和能力；旧数据缺失累计时间/步数保持null，累计和本轮进度分开。无可靠进程存活证据时不把磁盘记录冒充实时运行事实。
- 不生成假的稳定C编号、不把目录或排序位置当业务主键。旧产物可提供只读referenceId，持久checkpointId/alias和完整模型校验未具备时明确未知/未核验。
- 原始曲线分页/字段单位/缺口与缺失值保持；不把采样曲线当原始记录。旧记录没有时间戳就返回null，不用读取时间伪装评估生成时间。
- POST等控制请求拒绝；能力说明明确训练暂停、后台快照、指定模型续训、真实目录桥接等尚未在本模块接通。

## A-03：后续顺序与决定

- 首批读取合同和隔离测试通过后，再补checkpoint持久元数据/稳定别名和累计历史迁移。
- 训练暂停/继续、后台快照需要runner安全边界和独立操作状态；不得映射成预览暂停或旧stop-save。具体计时政策/恢复组件/快照捕获点必须按真实实现核对。
- UI原型的CSP和模拟数据保持，不在这一批切换生产；实际集成/部署与真实训练验收单独进行。
