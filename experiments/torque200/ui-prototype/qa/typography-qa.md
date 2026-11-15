# 字号与状态提示 QA

独立模拟页面 127.0.0.1:14159，未访问生产训练服务。IAB 不可用，使用既有独立 Edge 页面。截图 typography-before.jpg 与 typography-after.jpg 为 1440×900 同训练、10k 曲线场景；正文由 13px 增至 14px，更新后预览标题与状态分列。typography-mobile.jpg 为 390×844。

- 1440、900 视口文档宽分别为 1440、900；390 视口 clientWidth 与 scrollWidth 均 375，无页面横向溢出。
- 训练绿点 6px；累计 56.5% 时光效宽 526.7/931.7px。暂停、断线、延迟、准备、完成隐藏绿点和光效；后台快照保留；不限时铺满轨道。
- Value Loss 键盘 End 读出 0.300000 和环境步，选择对数后 data-scale=log；tooltip 12px。目录草稿弹窗可打开并说明不读取真实目录。
- 问题：390px 顶部“训练进行中”断成两行，末字单独占一行；已反馈。
- 未验：左右最小宽拖拽、页末遮挡、reduced-motion 模拟、console/CSP 及动画随时间位置；浏览器只读 evaluate 未提供 performance 对象，不能据此断言网络或日志无误。

- P2 修复复验：390px 顶部训练状态完整单行，checkpoint 与独立模拟位于第二行；clientWidth/scrollWidth 为 375/375。

## 补充复验

- 1440 桌面最小栏左 200px、右 230px。右栏动作接口原先截断；修复后标签上下排列，188px 选择框完整显示“腿位置＋轮速度”。
- 左栏 Value Loss 原先尺度控件覆盖标题；修复后控件在独立行，可切对数，标题及数值可见。tooltip 横向在栏内，但底部超出图表范围并覆盖相邻曲线标题，已反馈。截图 typography-narrow.jpg。
- 390px 滚到底参数区底 y=675.85，固定操作栏顶 y=703.45，间隔约 27.6px。
- 演示计时暂停时，300ms 扫光 left 78.25→148.03px，opacity 0.9；progress value 保持 848。
- 浏览器日志无页面 error/CSP；仅第三方扩展警告。reduced-motion 浏览器模拟未验。

- 最终浮层修复复验通过：左栏 200px，Value Loss 浮层底 539.65、下一标题顶 543.15；Policy Loss 浮层底 536.85、下一标题顶 540.35，均留约 3.5px。log/symlog 均可选择，title 和 aria-description 含范围、采样数及尺度理由。截图 typography-narrow.jpg、typography-policy-tooltip.jpg。
