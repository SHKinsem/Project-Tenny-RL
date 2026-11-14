# MCP 截图直接落盘

仅用于独立 Edge 插件已经返回的 PNG ImageContent；不是额外浏览器驱动。functions 内部处理 base64，不调用 text/notify 打印原始返回，也不把数据写入对话。Windows shell 前加载 powershell-safe-invocation。

在以下代码中替换 TAB_ID、ARCHIVE_DIR、SKILL_DIR 为实际观察和已解析的绝对路径；先创建归档的 images/ 目录。截图、保存和后续滚动必须串行。

```javascript
const r = await tools.mcp__edge_browser__edge_screenshot({tabId: TAB_ID});
store('taobao-screenshot', r); // 保存失败时 load 重用，避免再次请求浏览器
const png = r.content?.find(c => c.type === 'image' && c.mimeType === 'image/png');
if (!png?.data) throw Error('No PNG ImageContent; inspect small non-image metadata');
const encoded = png.data.replace(/\s+/g, '');
if (!/^[A-Za-z0-9+/=]+$/.test(encoded)) throw Error('Invalid base64');
const temp = ARCHIVE_DIR + '/images/01-overview.b64';
const output = ARCHIVE_DIR + '/images/01-overview.png';
const saved = await tools.apply_patch('*** Begin Patch\n*** Add File: ' + temp + '\n+' + encoded + '\n*** End Patch');
// 若 apply_patch 报错，停止并从 store 重试保存；不要打印 encoded。
const quote = s => "'" + s.replace(/'/g, "''") + "'";
const decoded = await tools.exec_command({
  cmd: ['python', quote(SKILL_DIR + '/scripts/save_screenshot.py'), quote(temp), quote(output)].join(' '),
  shell: 'pwsh.exe', max_output_tokens: 300
});
if (decoded.exit_code !== 0) throw Error(decoded.output);
text(output);
```

`save_screenshot.py` 校验 PNG 签名、保留原始字节，拒绝覆盖已有图，成功后删除自己读取的 .b64 临时文件。接着使用 view_image 查看生成的文件。多张图逐张更新编号；保存成功后再滚动，避免积累原始截图占用内存。若调用截图本身失败，只有错误信息，没有可恢复图像，按插件健康规则处理。
