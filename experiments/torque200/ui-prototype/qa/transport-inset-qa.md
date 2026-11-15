# 进度区内嵌折叠控件验收

独立模拟页，桌面 1440×900，打开高度 73.8px；summary x=4–32、28×28，正文从 x=40 开始，无独立标题行且未遮挡时间、进度或按钮。默认 marker 透明；键盘焦点 marker 可见，Enter 折叠稳定高度 28px，再 Enter 恢复 73.8px。累计进度始终 848/1500，开始禁用、暂停和保存可用。

390×844 下 clientWidth/scrollWidth 均 375，summary 同为 x=4–32；固定栏高 135px，页末底 675.95、栏顶 709，间隔约 33px。页面 error 日志为空。工具未提供 hover，未实测鼠标悬停。截图 transport-inset-desktop.jpg、transport-inset-mobile.jpg。

桌面截图已在视口稳定后重拍：innerWidth/clientWidth/scrollWidth 均 1440，右栏右缘约 1439.99，footer 右缘 1440。
