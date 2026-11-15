# 训练工作台：独立模拟原型

本轮UI、状态语义与交互验证的独立实现。当前使用内存数据，不接真实训练、渲染或文件系统；正常页面采用正式产品文案。正式前端见 [workbench-ui](../workbench-ui/README.md)，整体取舍见[设计约定](../../../docs/workbench-design.md)。

## 按需要阅读

| 需要做什么 | 阅读入口 |
| --- | --- |
| 理解本轮改进、设计原则和当前样式 | [DESIGN.md](DESIGN.md) |
| 对照原型字段、语义与原始后端需求 | [BACKEND-TODO.md](BACKEND-TODO.md)；实际接入状态见[后端说明](../../../docs/workbench-backend.md) |
| 运行采样、尺度、折叠和续训参考 | [examples/README.md](examples/README.md) |
| 查实际浏览器证据与未验范围 | [design-qa.md](design-qa.md)，最新进度/窗口证据见其首节；参数缩进见 [qa/indentation-20261115.md](qa/indentation-20261115.md) |
| 追溯最初审查或逐条用户修正 | [审查记录](../../../tasks/uiux-backlog.md)、[工作草稿](../../../tasks/uiux-workbench-draft.md) |

设计规范、运行方式、后端合同和历史证据各有自己的入口；后续更新对应文件，避免维护多份相互覆盖的规格。

## 本地运行

在仓库根目录运行，Python 3.11+，无需安装依赖：

```text
python experiments/torque200/ui-prototype/serve.py --no-browser
```

终端输出 `UI_PREVIEW_READY` 和自动分配的本机地址。服务只监听127.0.0.1，可用 `--port <端口>` 指定端口；关闭该终端只停止此预览。

也可在 PowerShell 启动隐藏窗口的独立预览，并将 PID 和日志保存在本目录 `.runtime/`：

```text
pwsh -NoProfile -File experiments/torque200/ui-prototype/start-preview.ps1
```

启动器输出 URL/PID 后等待服务退出；保留该执行会话供预览使用，不复用其他预览或训练服务。

普通URL不显示测试工具。使用 `?dev=1&scene=training&curve=dense-10k` 可打开开发场景与一万点样例；不加dev时仍可指定scene/curve，但不显示调试控件。刷新重置内存状态。

常用场景：`scene=resume-progress` 检查历史10分钟+追加10分钟从50%开始；`scene=disconnected` 检查断线；`curve=wide-loss` / `signed-loss` / `tiny-changes` 检查尺度与精度。完整场景集合以 [mock-engine.mjs](mock-engine.mjs) 的SCENARIOS为准。

README 展示图使用 `?scene=preview-running&curve=wide-loss`，不显示开发工具。`dense-10k` 刻意包含尖峰，用于检查异常点与密集采样，不能把它当作模型训练效果。

## 参数缩进

右侧同层大类的展开内容左缩进由16px增至24px（+8px），包括训练、控制、PPO、评估、视窗设置和详情。标题字号、右padding、边线/背景、箭头隐藏、折叠动效及主栏尺寸不变；收起时仍隐藏完整内容。实施与待验点见[缩进草稿](../../../tasks/ui-indentation-draft.md)。

## 隔离边界

- 页面无生产适配器、fetch/WebSocket或浏览器持久化，状态只存在内存。
- 静态服务仅GET/HEAD固定白名单；无目录索引、API代理、训练模块导入或子进程；写请求拒绝。
- CSP `connect-src 'none'` 禁止页面连接数据服务，脚本/样式为本地同源，无外部字体依赖。
- checkpoint目录入口只展示合成保存位置和内存选择，不打开Explorer、不读取manifest、不导入模型。
- dev=1只控制测试工具展示，不能启用真实接口。此原型未接生产；其他并行后端的完成情况需按交接清单核对。

## 验证

离线语义参考：

```text
node experiments/torque200/ui-prototype/examples/reference-scenarios.mjs
```

模块与静态服务检查：

```text
node --test experiments/torque200/ui-prototype/mock-engine.test.mjs experiments/torque200/ui-prototype/layout.test.mjs experiments/torque200/ui-prototype/chart-model.test.mjs
python -m unittest discover -s experiments/torque200/ui-prototype -p test_preview_server.py -v
node --check experiments/torque200/ui-prototype/app.mjs
```

也可运行 `pwsh -NoProfile -File experiments/torque200/ui-prototype/verify-isolated.ps1`；该脚本把HTTP测试的临时目录限定在本副本 `.runtime/test-temp/`。HTTP用例只启动自己的临时回环服务；某些Windows沙箱会限制回环连接，应在有回环权限的同用户环境验证，不能据连接权限错误判定页面逻辑失败。不要为测试复用正在训练的服务。

浏览器简单点按与窄面板验收交给Terra；浏览器验证与单测分别记录。实际hover完整路径、连续拖动阈值微回退、真实触屏、200%缩放/屏幕阅读器、reduced-motion及逐帧表现仍需专项验收，不能从单测或截图推断已通过。

## 数学与浏览器能力参考

- [D3正值对数](https://d3js.org/d3-scale/log)、[对称对数](https://d3js.org/d3-scale/symlog)：本原型使用本地实现，没有加载D3。
- [MDN ::marker](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Selectors/::marker)：原生折叠标记的颜色过渡。
- [MDN showDirectoryPicker](https://developer.mozilla.org/en-US/docs/Web/API/Window/showDirectoryPicker)：此模拟原型未调用；正式目录能力见工作台控制合同。
