let refreshTimer0_;
let receiveData = false;
// const onedatalen = 22; // 将 onedatalen 作为 JavaScript 变量传递
let selectedColumns = [2, 4, 6]; // 默认选择的列

document.getElementById('receiveData').addEventListener('change', function() {
  if (this.checked && !receiveData) {
    clearHistory(); // 清空历史记录
  }
  receiveData = this.checked;
  toggleAutoDownload(); // 当 receiveData 状态改变时调用 toggleAutoDownload
});

function submitForm() {
  fetch('/setK', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(new FormData(document.getElementById('kForm'))).toString()
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('otherString0_').innerText = data.otherString0_;
    var historyList = document.getElementById('otherStringHistory');
    var newItem = document.createElement('li');
    newItem.innerText = data.otherString;
    historyList.appendChild(newItem);
    refreshMatrix();
    alert(data.message);
  });
}

// 添加生成和提交 MatrixXf 矩阵的 JavaScript 函数
function generateMatrixTable(matrix) {
  const table = document.getElementById('matrixTable');
  table.innerHTML = '';
  for (let i = 0; i < matrix.length; i++) {
    const row = document.createElement('tr');
    for (let j = 0; j < matrix[i].length; j++) {
      const cell = document.createElement('td');
      cell.innerHTML = `<input type='text' name='M${i}_${j}' value='${matrix[i][j]}'>`;
      row.appendChild(cell);
    }
    table.appendChild(row);
  }
}

function submitMatrixForm() {
  fetch('/setMatrix', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(new FormData(document.getElementById('matrixForm'))).toString()
  })
  .then(response => response.json())
  .then(data => {
    alert(data.message);
  });
}

function refreshMatrix() {
  fetch('/getMatrix')
  .then(response => response.json())
  .then(data => {
    generateMatrixTable(data.matrix);
  });
}

refreshMatrix(); // 每1000ms刷新一次 MatrixXf

function generateVectorTable(index, vector, labels) {
  const container = document.getElementById('vectorsContainer');
  const form = document.createElement('form');
  form.id = 'vectorForm' + index;
  const table = document.createElement('table');
  table.id = 'vectorTable' + index;
  const row = document.createElement('tr');
  for (let i = 0; i < vector.length; i++) {
    const cell = document.createElement('td');
    cell.innerHTML = `[${i}]${labels[i]}: <input type='text' name='V${index}_${i}' value='${vector[i]}'>`;
    row.appendChild(cell);
  }
  table.appendChild(row);
  form.appendChild(table);
  const submitButton = document.createElement('button');
  submitButton.type = 'button';
  submitButton.innerText = 'Submit Vector' + index;
  submitButton.onclick = function() { submitVectorForm(index); };
  form.appendChild(submitButton);
  container.appendChild(form);
}

function refreshVector(index) {
  fetch('/getVector' + index)
  .then(response => response.json())
  .then(data => {
    generateVectorTable(index, data.vector, data.labels);
  });
}

function submitVectorForm(index) {
  fetch('/setVector' + index, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(new FormData(document.getElementById('vectorForm' + index))).toString()
  })
  .then(response => response.json())
  .then(data => {
    alert(data.message);
  });
}

function loadAllVectors() {
  fetch('/getVectorSize')
  .then(response => response.json())
  .then(data => {
    for (let i = 0; i < data.size; i++) {
      refreshVector(i);
    }
  });
}

// 加载所有现有的 VectorXf 表单
loadAllVectors();

function refreshOtherString0_() {
  fetch('/getOtherString0_')
  .then(response => response.json())
  .then(data => {
    if (data.otherString0_.length > 0) { // 如果 otherString0_ 不是空
      document.getElementById('otherString0_').innerText = data.otherString0_;
    }
  });
}

function refreshOtherData_json() {
  if (!receiveData) return;
  fetch('/getOtherData')
  .then(response => response.json()) // 解析JSON格式的数据
  .then(data => {
    var historyList = document.getElementById('otherStringHistory');
    data.forEach(itemArray => {
      var item = itemArray.map(value => parseFloat(value.toFixed(6))); // 使用 parseFloat 去除多余的零
      var newItem = document.createElement('li');
      newItem.innerText = item.join(', '); // 使用 join 方法将数组元素连接成字符串
      historyList.appendChild(newItem);
    });
    document.getElementById('historyCount').innerText = historyList.getElementsByTagName('li').length;
    scrollToBottom(); // 滚动到最底部
    // updateChart(data); // 更新图表数据
  });
}

function refreshOtherData() {
  if (!receiveData) return;
  fetch('/getOtherData')
  .then(response => response.arrayBuffer()) // 解析二进制数据
  .then(buffer => {
    var data = [];
    var view = new DataView(buffer);
    var floatSize = 4; // 每个 float 占用 4 个字节
    var dataSizeFloat = view.getFloat32(0, true); // 获取前四个字节，表示 otherData[0].size()，使用小端字节序
    var dataSize = Math.floor(dataSizeFloat); // 将 float 转换为 int
    for (var i = floatSize; i < buffer.byteLength; i += floatSize * dataSize) {
      var itemArray = [];
      for (var j = 0; j < dataSize; j++) {
        var value = view.getFloat32(i + j * floatSize, true); // 读取 float 值，使用小端字节序
        itemArray.push(value);
      }
      data.push(itemArray);
    }

    var historyList = document.getElementById('otherStringHistory');
    data.forEach(itemArray => {
      var item = itemArray.map(value => parseFloat(value.toFixed(6))); // 使用 parseFloat 去除多余的零
      var newItem = document.createElement('li');
      newItem.innerText = item.join(', '); // 使用 join 方法将数组元素连接成字符串
      historyList.appendChild(newItem);
    });
    document.getElementById('historyCount').innerText = historyList.getElementsByTagName('li').length;
    scrollToBottom(); // 滚动到最底部
    // updateChart(data); // 更新图表数据
  });
}

function copyHistory() {
  var historyList = document.getElementById('otherStringHistory');
  var historyItems = historyList.getElementsByTagName('li');
  var historyText = '';
  for (var i = 0; i < historyItems.length; i++) {
    historyText += historyItems[i].innerText + '\n';
  }
  var tempTextArea = document.createElement('textarea');
  tempTextArea.value = historyText;
  document.body.appendChild(tempTextArea);
  tempTextArea.select();
  document.execCommand('copy');
  document.body.removeChild(tempTextArea);
  alert('History copied to clipboard');
}

function setRefreshInterval0_() {
  clearInterval(refreshTimer0_);
  const interval0_ = document.getElementById('refreshInterval0_').value;
  refreshTimer0_ = setInterval(refreshOtherString0_, interval0_);
}

function saveHistory() {
  var historyList = document.getElementById('otherStringHistory');
  var historyItems = historyList.getElementsByTagName('li');
  if (historyItems.length === 0) {
    return; // 如果历史记录为空，则不进行保存操作
  }
  var historyText = '';
  for (var i = 0; i < historyItems.length; i++) {
    historyText += historyItems[i].innerText + '\n';
  }
  var blob = new Blob([historyText], { type: 'text/plain' });
  var link = document.createElement('a');
  link.href = window.URL.createObjectURL(blob);
  link.download = 'history.txt';
  link.click();
  historyList.innerHTML = ''; // 清空历史记录
  document.getElementById('historyCount').innerText = '0'; // 更新条数
}

function toggleAutoDownload() {
  autoDownloadEnabled = document.getElementById('autoDownload').checked && receiveData;
  const interval = document.getElementById('autoDownloadInterval').value;
  document.getElementById('autoDownload').disabled = !receiveData; // 当 receiveData 为 false 时禁用 autoDownload
  document.getElementById('autoDownloadInterval').disabled = !autoDownloadEnabled;
  if (autoDownloadEnabled) {
    autoDownloadTimer = setInterval(saveHistory, interval); // 根据选择的间隔时间自动下载
  } else {
    clearInterval(autoDownloadTimer);
  }
}

function clearHistory() {
  var historyList = document.getElementById('otherStringHistory');
  historyList.innerHTML = ''; // 清空历史记录
  document.getElementById('historyCount').innerText = '0'; // 更新条数
}

function scrollToBottom() {
  var historyContainer = document.getElementById('historyContainer');
  historyContainer.scrollTop = historyContainer.scrollHeight;
}

function groupKValues() {
  const groups = [
    ['Wheel Control', ['wheel_ctrl_scale', 'wheel_Kp', 'wheel_Ki', 'wheel_Kd', 'wheel_vel_smoothing_alpha(0,1)']],
    ['Thigh Control', ['thigh_Kp', 'thigh_Kd', 'thigh_smoothing_alpha(0,1)']],
    ['Attitude & Observation', ['sinwheel_scale', 'sinpitch_scale', 'sinyaw_scale', 'pitch_max', 'pitch_zero_low', 'pitch_zero_high', 'sinroll_scale']],
    ['Position / Heading Hold', ['hold_error_clip_enable', 'hold_error_clip_ds', 'hold_error_clip_dpsi']],
    ['Commands', ['cmd_lin_vel_x(-0.2,0.2)', 'cmd_ang_vel_yaw(-1.5,1.5)', 'cmd_height_value(-0.5,0)', 'cmd_roll_value(-0.2,0.2)', 'command_vx_rate']],
    ['System & Filtering', ['send_interval', 'run_interval', 'control_smooth(0, 1)']]
  ];
  const descriptions = {
    'wheel_ctrl_scale': '轮输出缩放',
    'wheel_Kp': '轮速比例增益',
    'wheel_Ki': '轮速积分增益',
    'wheel_Kd': '轮速微分增益',
    'sinwheel_scale': '当前未使用',
    'sinpitch_scale': '俯仰观测缩放',
    'sinyaw_scale': '航向观测缩放',
    'hold_error_clip_enable': '启用误差限幅',
    'hold_error_clip_ds': '位置误差半幅',
    'hold_error_clip_dpsi': '航向误差半幅',
    'pitch_max': '俯仰保护阈值',
    'send_interval': '调试发送间隔',
    'run_interval': '策略调用间隔',
    'thigh_Kp': '大腿比例增益',
    'thigh_Kd': '大腿微分增益',
    'thigh_smoothing_alpha(0,1)': '大腿输出平滑',
    'wheel_vel_smoothing_alpha(0,1)': '轮速观测平滑',
    'control_smooth(0, 1)': '轮输出平滑',
    'cmd_lin_vel_x(-0.2,0.2)': '前后速度指令',
    'cmd_ang_vel_yaw(-1.5,1.5)': '偏航速度指令',
    'cmd_height_value(-0.5,0)': '机体高度指令',
    'cmd_roll_value(-0.2,0.2)': '横滚目标指令',
    'command_vx_rate': '速度变化率',
    'pitch_zero_low': '下蹲俯仰零点',
    'pitch_zero_high': '站立俯仰零点',
    'sinroll_scale': '横滚观测缩放'
  };
  const container = document.getElementById('kValues');
  const items = Array.from(container.querySelectorAll('.k-item'));
  for (const [title, keys] of groups) {
    const section = document.createElement('section');
    section.className = 'k-group';
    const heading = document.createElement('h2');
    heading.innerText = title;
    section.appendChild(heading);
    const content = document.createElement('div');
    content.className = 'k-group-items';
    for (const key of keys) {
      const item = items.find(element => element.dataset.kName === key);
      if (item) {
        item.querySelector('.k-desc').innerText = descriptions[key] || '';
        content.appendChild(item);
      }
    }
    section.appendChild(content);
    container.appendChild(section);
  }
}

groupKValues();

refreshTimer0_ = setInterval(refreshOtherString0_, 1000); // 初始化刷新间隔
setInterval(refreshOtherData, 1000); // 每1000ms刷新一次 otherData

function goToControlPage() {
  // // 发送当前页面信息到服务器
  // fetch('/savePageState', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  //   body: 'currentPage=control'
  // }).then(() => {
    // 页面跳转
    window.location.href = 'control.html';
  // });
}

