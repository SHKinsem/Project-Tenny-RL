function goToMainPage() {
  // 发送当前页面信息到服务器
  // fetch('/savePageState', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  //   body: 'currentPage=main'
  // }).then(() => {
    // 页面跳转
    window.location.href = 'main.html';
  // });
}

function loadControlPageData() {
  fetch('/getControlPageData')
    .then(response => response.json())
    .then(data => {
      document.getElementById('height_bar').value = data.height_bar;
      document.getElementById('height_value').innerText = data.height_bar;
      document.getElementById('oscillation_bar').value = data.oscillation_bar;
      document.getElementById('oscillation_value').innerText = data.oscillation_bar;
      document.getElementById('toggleSwitch').checked = data.switcher;
      document.getElementById('switchState').innerText = data.switcher ? "ON" : "OFF";
    })
    .catch(error => console.error('Error loading control page data:', error));
}

function loadImuData() {
  fetch('/getImuData')
    .then(response => response.json())
    .then(data => {
      document.getElementById('imuYaw').innerText = data.yaw_rad.toFixed(4);
      document.getElementById('imuPitch').innerText = data.pitch_rad.toFixed(4);
      document.getElementById('imuRoll').innerText = data.roll_rad.toFixed(4);
    })
    .catch(error => console.error('Error loading IMU data:', error));
}

setInterval(loadImuData, 1000);

let controlSocket;
let controlSocketReconnect;
let switcherRestored = false;
let latencyPingStartedAt = 0;
let latencyPingInFlight = false;

function setConnectionStatus(mode, rtt = null) {
  const modeElement = document.getElementById('connectionMode');
  modeElement.innerText = mode;
  modeElement.dataset.state = mode === 'WebSocket' ? 'online' : 'fallback';
  document.getElementById('connectionLatency').innerText = rtt === null ? '--' : `${rtt.toFixed(1)} ms`;
}

function startLatencyProbe() {
  if (controlSocket?.readyState !== WebSocket.OPEN) {
    return;
  }
  latencyPingStartedAt = performance.now();
  latencyPingInFlight = true;
  controlSocket.send('P');
}

function connectControlSocket() {
  setConnectionStatus('Connecting...');
  controlSocket = new WebSocket(`ws://${location.hostname}:81/`);
  controlSocket.onopen = () => {
    setConnectionStatus('WebSocket');
    restoreSwitcher();
    startLatencyProbe();
  };
  controlSocket.onmessage = (event) => {
    if (event.data === 'P' && latencyPingInFlight) {
      latencyPingInFlight = false;
      setConnectionStatus('WebSocket', performance.now() - latencyPingStartedAt);
    }
  };
  controlSocket.onclose = () => {
    latencyPingInFlight = false;
    setConnectionStatus(httpFallbackAllowed() ? 'HTTP fallback' : 'WS disconnected');
    clearTimeout(controlSocketReconnect);
    controlSocketReconnect = setTimeout(connectControlSocket, 1000);
  };
  controlSocket.onerror = () => controlSocket.close();
}

function sendControlFrame(frame) {
  if (controlSocket?.readyState !== WebSocket.OPEN) {
    return false;
  }
  controlSocket.send(frame);
  return true;
}

function restoreSwitcher() {
  if (switcherRestored) {
    return;
  }
  switcherRestored = true;
  const state = localStorage.getItem('controlSwitcher');
  if (state === null) {
    return;
  }
  if (sendControlFrame(`S:${state}`)) {
    return;
  }
  if (!httpFallbackAllowed()) {
    return;
  }
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/switchToggle', true);
  xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
  xhr.send(`state=${state}`);
}

function httpFallbackAllowed() {
  return document.getElementById('httpFallback').checked;
}

function httpFallbackChanged() {
  if (controlSocket?.readyState === WebSocket.OPEN) {
    return;
  }
  setConnectionStatus(httpFallbackAllowed() ? 'HTTP fallback' : 'WS disconnected');
}

setInterval(startLatencyProbe, 1000);

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('httpFallback').checked = location.port !== '';
  connectControlSocket();
  restoreSwitcher();
  loadControlPageData();
  loadImuData();
  layoutJoystick();
  drawJoystick();
});

const canvas = document.getElementById('joystick');
const ctx = canvas.getContext('2d');

let centerX = 0;
let centerY = 0;
let maxRadius = 0;
let joystickX = 0;
let joystickY = 0;
let stickX = 0;
let stickY = 0;
let isDragging = false;

function layoutJoystick() {
  const css = canvas.getBoundingClientRect();
  const side = Math.max(1, Math.round(Math.min(css.width || 0, css.height || 0) || canvas.clientWidth || 300));
  if (canvas.width !== side || canvas.height !== side) {
    canvas.width = side;
    canvas.height = side;
  }
  centerX = canvas.width / 2;
  centerY = canvas.height / 2;
  maxRadius = canvas.width / 2 * 0.8;
  if (!isDragging) {
    joystickX = centerX;
    joystickY = centerY;
    stickX = 0;
    stickY = 0;
  }
}

function padGeom() {
  const rect = canvas.getBoundingClientRect();
  return {
    cx: rect.left + rect.width / 2,
    cy: rect.top + rect.height / 2,
    maxR: Math.min(rect.width, rect.height) / 2 * 0.8,
  };
}

function showStick(nx, ny) {
  stickX = nx;
  stickY = ny;
  joystickX = centerX + nx * maxRadius;
  joystickY = centerY - ny * maxRadius;
  document.getElementById('joystickX').innerText = nx.toFixed(2);
  document.getElementById('joystickY').innerText = ny.toFixed(2);
  drawJoystick();
}

function applyStickFromClient(clientX, clientY) {
  layoutJoystick();
  const p = padGeom();
  const dx = clientX - p.cx;
  const dy = clientY - p.cy;
  const dist = Math.min(p.maxR, Math.hypot(dx, dy));
  const ang = Math.atan2(dy, dx);
  // 可见圆心为原点。屏幕向下 → 发给固件的 y 为负 = 后退。
  const nx = p.maxR > 0 ? (dist * Math.cos(ang)) / p.maxR : 0;
  const ny = p.maxR > 0 ? -(dist * Math.sin(ang)) / p.maxR : 0;
  showStick(nx, ny);
  sendJoystickMove(nx, ny);
}

function resetStick() {
  isDragging = false;
  showStick(0, 0);
  sendJoystickMove(0, 0, true);
}

layoutJoystick();
window.addEventListener('resize', () => {
  layoutJoystick();
  drawJoystick();
});

// 固件 1s 无更新会把摇杆清零。纯上/下时 x≈0，必须按幅度续发，不能要求 x、y 都非零。
setInterval(() => {
  if (!isDragging) {
    return;
  }
  if (Math.hypot(stickX, stickY) > 0.05) {
    sendJoystickMove(stickX, stickY);
  }
}, 100);

function updateSliderValues() {
  document.getElementById('oscillation_value').innerText = document.getElementById('oscillation_bar').value;
  document.getElementById('height_value').innerText = document.getElementById('height_bar').value;
}

function sendSliderValue(url, value) {
  const frame = url === '/setHeight' ? `H:${value}` : `O:${value}`;
  if (sendControlFrame(frame)) {
    return;
  }
  if (!httpFallbackAllowed()) {
    return;
  }
  const startedAt = performance.now();
  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `value=${value}`
  })
  .then(response => {
    setConnectionStatus('HTTP fallback', performance.now() - startedAt);
    return response.text();
  })
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
}

document.getElementById('oscillation_bar').addEventListener('input', function() {
  const value = this.value;
  document.getElementById('oscillation_value').innerText = value;
  sendSliderValue('/setOscillation', value);
});

document.getElementById('height_bar').addEventListener('input', function() {
  const value = this.value;
  document.getElementById('height_value').innerText = value;
  sendSliderValue('/setHeight', value);
});


function drawJoystick() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 绘制外圈
  ctx.beginPath();
  ctx.arc(centerX, centerY, maxRadius, 0, Math.PI * 2, true);
  ctx.fillStyle = 'rgba(200, 200, 200, 0.5)';
  ctx.fill();
  ctx.strokeStyle = '#999';
  ctx.stroke();

  // 绘制十字线
  ctx.beginPath();
  ctx.moveTo(centerX - maxRadius, centerY);
  ctx.lineTo(centerX + maxRadius, centerY);
  ctx.moveTo(centerX, centerY - maxRadius);
  ctx.lineTo(centerX, centerY + maxRadius);
  ctx.strokeStyle = '#666'; // 十字线颜色
  ctx.lineWidth = 1; // 十字线宽度
  ctx.stroke();

  // 绘制中心点到摇杆的线
  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(joystickX, joystickY);
  const lineGradient = ctx.createLinearGradient(centerX, centerY, joystickX, joystickY);
  lineGradient.addColorStop(0, '#000');
  lineGradient.addColorStop(1, '#ff0000');
  ctx.strokeStyle = lineGradient; // 线条颜色渐变
  ctx.lineWidth = 2; // 线条宽度
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
  ctx.shadowBlur = 4;
  ctx.stroke();
  ctx.shadowBlur = 0; // 重置阴影

  // 绘制中间的摇杆
  ctx.beginPath();
  ctx.arc(joystickX, joystickY, maxRadius * 0.25, 0, Math.PI * 2, true); // 增加中间圆的半径
  const gradient = ctx.createRadialGradient(joystickX, joystickY, maxRadius * 0.1, joystickX, joystickY, maxRadius * 0.25);
  gradient.addColorStop(0, '#ff0000');
  gradient.addColorStop(1, '#800000');
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.strokeStyle = '#660000';
  ctx.stroke();
}

let lastJoystickSentAt = 0;

function sendJoystickMove(x, y, force = false) {
  const now = performance.now();
  if (!force && now - lastJoystickSentAt < 20) {
    return;
  }
  lastJoystickSentAt = now;
  if (sendControlFrame(`J:${x.toFixed(4)},${y.toFixed(4)}`)) {
    return;
  }
  if (!httpFallbackAllowed()) {
    return;
  }
  const startedAt = performance.now();
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/joystickMove", true);
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.onload = () => setConnectionStatus('HTTP fallback', performance.now() - startedAt);
  xhr.send(`x=${x}&y=${y}`);
}

function toggleSwitchChanged() {
  const state = document.getElementById('toggleSwitch').checked;
  localStorage.setItem('controlSwitcher', String(state));
  // 更新显示的状态值
  document.getElementById('switchState').innerText = state ? "ON" : "OFF";
  if (sendControlFrame(`S:${state}`)) {
    return;
  }
  if (!httpFallbackAllowed()) {
    return;
  }
  const startedAt = performance.now();
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/switchToggle", true);
  xhr.onload = () => setConnectionStatus('HTTP fallback', performance.now() - startedAt);
  xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
  xhr.send(`state=${state}`);
}

canvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const p = padGeom();
  if (Math.hypot(e.clientX - p.cx, e.clientY - p.cy) > p.maxR) {
    return;
  }
  isDragging = true;
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch (err) {
    // 旧环境没有 pointer capture 时仍靠 pointermove。
  }
  applyStickFromClient(e.clientX, e.clientY);
});

canvas.addEventListener('pointermove', (e) => {
  if (!isDragging) {
    return;
  }
  e.preventDefault();
  applyStickFromClient(e.clientX, e.clientY);
});

function endStick(e) {
  if (!isDragging) {
    return;
  }
  e.preventDefault();
  resetStick();
}

canvas.addEventListener('pointerup', endStick);
canvas.addEventListener('pointercancel', endStick);
canvas.style.touchAction = 'none';

drawJoystick();