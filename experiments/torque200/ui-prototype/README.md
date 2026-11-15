# 训练工作台：独立模拟原型

本目录保留工作台的内存数据与交互，作为布局、动效及曲线压力场景的独立参考。没有接入真实训练、渲染、文件系统或后端 API；正式前端见 [workbench-ui](../workbench-ui/README.md)，整体取舍见[设计约定](../../../docs/workbench-design.md)。

## 本地运行

在本 Worktree 仓库根目录运行，Python 3.11+，无需安装依赖：

```text
python experiments/torque200/ui-prototype/serve.py --no-browser
```

终端输出 `UI_PREVIEW_READY` 和自动分配的新本机端口。服务只监听127.0.0.1；不复用其他正在运行的预览或训练服务。

也可在 PowerShell 启动隐藏窗口的独立预览，并将 PID 和日志保存在本目录 `.runtime/`：

```text
pwsh -NoProfile -File experiments/torque200/ui-prototype/start-preview.ps1
```

启动器输出 URL/PID 后等待服务退出；保留该执行会话供预览使用。

正常URL不显示测试工具。`?dev=1&scene=training&curve=dense-10k` 可显示测试场景；常用 `scene=resume-progress`、`scene=disconnected` 及 `curve=wide-loss`。这些查询参数都只影响内存模拟，刷新重置状态。

README 展示图使用 `?scene=preview-running&curve=wide-loss`，不显示开发工具。`dense-10k` 刻意包含尖峰，用于检查异常点与密集采样，不能把它当作模型训练效果。

## 本轮视觉范围

右侧同层大类的展开内容左缩进由16px增至24px（+8px），包括训练、控制、PPO、评估、视窗设置和详情。标题字号、右padding、边线/背景、箭头隐藏、折叠动效及主栏尺寸不变；收起时仍隐藏完整内容。

实施与待验点见 [缩进草稿](../../../tasks/ui-indentation-draft.md)。真实后端合同与开发状态见 [后端说明](../../../docs/workbench-backend.md)。此副本不搬入原树的历史截图或QA记录。

## 验证

```text
node --test experiments/torque200/ui-prototype/mock-engine.test.mjs experiments/torque200/ui-prototype/layout.test.mjs experiments/torque200/ui-prototype/chart-model.test.mjs
python -m unittest discover -s experiments/torque200/ui-prototype -p test_preview_server.py -v
node --check experiments/torque200/ui-prototype/app.mjs
```

也可运行 `pwsh -NoProfile -File experiments/torque200/ui-prototype/verify-isolated.ps1`；该脚本把HTTP测试的临时目录限定在本副本 `.runtime/test-temp/`。Windows沙箱若限制回环连接，请在有回环权限的同用户环境验证，不能据连接权限错误判定页面逻辑失败。

## 隔离

页面无生产适配器、fetch/WebSocket或浏览器持久化。静态服务仅GET/HEAD白名单，无API代理、训练模块导入和子进程；写请求拒绝。CSP保留 `connect-src 'none'` 及本地同源脚本/样式。checkpoint目录仅为内存交互，不打开Explorer或读取真实manifest。

浏览器简单点按与窄面板验收交给Terra，离线模块与服务测试不能代替视觉验收。
