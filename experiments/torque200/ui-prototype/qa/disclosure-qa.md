# 折叠标题动效验收

独立模拟页，Edge，1440×900。训练标题键盘焦点前 marker rgba(0,0,0,0)，焦点稳定后 rgb(150,150,150)，计算 transitionDuration 0.16s；标签 x 从 1173.44 到 1185.44（12px），summary、元数据和输入位置不变。截图 disclosure-rest.jpg、disclosure-hover.jpg（后者为键盘焦点，非鼠标悬停）。单窗口组与窗口 marker 默认透明。

390px clientWidth/scrollWidth 均 375，固定栏高度 153px；页面 error 日志为空。浏览器工具未提供鼠标悬停方法，未实测 hover；reduce-motion 未模拟；未完整验证多窗口及键盘开闭稳定结果。
