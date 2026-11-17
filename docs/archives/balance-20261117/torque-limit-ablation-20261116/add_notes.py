import json
from pathlib import Path

root=Path(__file__).parent/'comparison'
note=('本轮±2组0步和训练后的累计合格时间都为0。两组“相对各自0步增量之差”出现正数，'
      '来自±1组下降而±2组已处于0的下限，不能解释成±2组学得更好。'
      '本轮检验扩大动作范围并沿用同一BC参数的方案，尚未检验物理输出等价的初始化。')
summary=json.loads((root/'paired-summary.json').read_text())
summary['result_reading_notes']=[note]
(root/'paired-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
path=root/'report.html';html=path.read_text(encoding='utf-8')
marker='<h2>逐种子配对差</h2>'
if '本轮数据注意' not in html:
    html=html.replace(marker,'<p><strong>本轮数据注意：</strong>'+note+'</p>'+marker)
    path.write_text(html,encoding='utf-8')
