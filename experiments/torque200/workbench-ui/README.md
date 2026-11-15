# 真实训练工作台

独立于 `ui-prototype` 的正式前端副本。根页面由现有训练预览服务提供，静态模块统一位于 `/workbench/`；旧页面保留在 `/legacy`。

页面载入不会启动训练；只有用户操作才提交真实命令。本目录不包含模拟引擎或合成曲线，也不维护机器人模式表。模式来自 `/api/workbench/v1/catalog`，保存描述来自 `/api/workbench/v1/descriptor`。缺失内容显示未知。

- `adapter.mjs`：纯数据边界。监督器决定活动状态与最近模型续训资格；累计记录及保存描述须匹配当前模式与运行身份。
- `transport.mjs`：独立状态心跳与单一辅助通道，最多两个API连接；学习窗口与历史列表低频更新。辅助慢读不阻塞心跳，失败保留数据、禁写并自动退避重连。
- `app.mjs`：参数草稿、精确模型选择、训练控制、目录桥接及至多四个真实预览窗口。
- `layout.mjs`、`chart-model.mjs`、`chart-view.mjs`：折叠、拖拽、细线图及原始点查看。

能力由 `/capabilities` 的 `legacyControls` 决定。训练暂停与手动保存还要求当前进程 `workbench_control_protocol=1`；旧进程保持禁用，不能为补能力重启用户训练。保存请求接受后等待 `workbench_save_results` 的 completed/failed 回执，不能把 HTTP 成功当作保存完成。

历史起点来自 `/checkpoint-options` 的兼容验证结果，提交所选精确 path。动态 `/parameters` 表单只提交编辑过的字段；空对象保持 auto 与继承默认。训练中锁定，轮询不覆盖正在编辑的草稿。恢复网络与归一化，优化器是否恢复以接口合同为准。

每窗拥有独立的同步、加载、暂停、重置和释放。首帧阶段即挂载流以建立客户端，载入条等 ready 才退出；关闭意图不会被迟到响应重新开启。加载有真实分母时显示真实进度，未知时只显示阶段及不定动画。

目录选择/打开调用 `/api/local/directories`；选择取消不写配置，保存目录经 `/storage` 验证，导入经 `/import` 完整复制及兼容检查。原生选择等待上限 310 秒、导入等待上限 130 秒；两者不阻塞状态轮询。

在仓库根目录运行：

```powershell
node --test experiments/torque200/workbench-ui/adapter.test.mjs
node --check experiments/torque200/workbench-ui/app.mjs
node --check experiments/torque200/workbench-ui/transport.mjs
python -m unittest discover -s tests -p test_workbench_preview_http.py
```

本机内存验收服务、慢加载/取消/首帧失败示例与说明保存在 `tasks/results/workbench-completion-ui/`，属于忽略的本机测试产物，不随仓库发布。可移植的边界与HTTP回归位于本目录测试和仓库tests。部署由现有训练服务的静态页面分发负责。
