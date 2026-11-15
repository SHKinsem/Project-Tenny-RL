# 工作台动效增量 QA

2026-11-15，Edge 独立 14159 模拟页面；不连接真实训练。1440×900 单窗口 `viewport-grid` 高约 534.6 px，四窗口截图显示四个空状态充分占用中央区域；九窗及四窗切换可见。三主栏和顶部、底部、一个曲线子组可收起恢复；分隔条拖拽和方向键均改变宽度，模型弹窗 Escape 后焦点回到来源按钮。续训初始 progress value=600、max=1200，暂停时模拟保存后 checkpoint 更新且仍暂停；截图中的校验为演示。

390×844 预览可折叠和展开，底部操作可见。页面 error 日志为空，未见 CSP 错误。截图 `workbench-motion-desktop.jpg`、`workbench-motion-resume.jpg`、`workbench-motion-mobile.jpg` 已保存并查看。浏览器接口未提供 reduced-motion 模拟；未测试该模式。快速连击的自动化定位未匹配，未完成快速点击竞争测试；未对动效帧率或所有子组逐项验收。结束恢复训练场景、时钟暂停、主栏和四窗展开，视口覆盖已复位。
