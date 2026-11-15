# 离线可执行参考

本目录保存可复用的确定性示例，直接调用原型已有模块，不复制采样、尺度、布局或训练状态算法。
仅用于本机离线理解与检查；产品页面不导入此脚本，也不增加演示提示。

## 运行

在仓库根目录使用本机 Node.js，无需安装依赖：

```text
node experiments/torque200/ui-prototype/examples/reference-scenarios.mjs
```

成功时退出码为 0，并输出四组精简 JSON；少量内置断言检查关键语义，失败会直接报错。

## 输出含义

1. `sampling`：10000 个原始点在 160、280、640 像素宽度下的线条采样数量。
   `unsampledHover` 读取没有参与绘线的原始记录，保留步数、原值及六位有效数字显示。
2. `scales`：跨数量级正 Loss 使用 Log；含零和负值的 Policy Loss 使用 SymLog，显式请求 Log 也安全回退。
   原值不取绝对值、不补 epsilon；`tinyChanges` 保留约 1e-9 的细微变化及科学计数显示。
3. `paneGesture`：左栏原始拖动宽度在 160px 收起，逆向回到 223px 仍保持收起，达到 224px 才展开。
   另示范释放后从 48px 窄边重新拖回；右栏不变，三栏宽度始终合计 1424px。
4. `resumeProgress`：`cumulativePercent` 是历史与本轮合计占 `cumulativeTargetSeconds` 的比例，依次为 50% → 55% → 暂停仍 55% → 60%。
   `segmentPercent` 单独表示本轮 elapsed/budget，依次为 0% → 10% → 暂停仍 10% → 20%，不能与累计占比混用。
   `unknownHistory` 的累计目标和累计占比保留 `null`，不把未知历史当作零；已知本轮计时与预算时，`segmentPercent` 仍可计算。

## 实际代码入口

- [chart-model.mjs](../chart-model.mjs)：曲线采样、原始点查询、尺度与数值格式。
- [layout.mjs](../layout.mjs)：纯尺寸与带滞回的拖拽手势模型。
- [mock-engine.mjs](../mock-engine.mjs)：现有场景、继续/暂停及时间推进状态。

脚本只导入这些本地实现和 Node 内置断言；不启动浏览器、服务、训练或外部进程，不联网、不读写结果文件。
所有输入均为内存合成数据，时间固定；它验证离线交互语义，不代表真实后端、模型训练效果或生产集成已通过。
