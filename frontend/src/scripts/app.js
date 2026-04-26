// SmileOrNot — camera capture + adaptive predict loop + canvas overlay.

const STATE = { IDLE: 'idle', STARTING: 'starting', RUNNING: 'running', ERROR: 'error' };
let state = STATE.IDLE;
let stream = null;
let abortController = null;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const toggleBtn = document.getElementById('toggle');

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

async function start() {
  state = STATE.STARTING;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    sizeOverlayToVideo();
    state = STATE.RUNNING;
    toggleBtn.textContent = 'Stop Camera';
    loop();
  } catch (e) {
    state = STATE.ERROR;
    showStatus(e.name === 'NotAllowedError' ? 'Camera permission denied.' : 'Camera unavailable.');
  }
}

function stop() {
  state = STATE.IDLE;
  if (abortController) abortController.abort();
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  toggleBtn.textContent = 'Start Camera';
  showStatus('Stopped.');
}

async function loop() {
  while (state === STATE.RUNNING) {
    try {
      const blob = await captureFrame();
      abortController = new AbortController();
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      const r = await fetch('/predict', { method: 'POST', body: fd, signal: abortController.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const { boxes, inference_ms } = await r.json();
      if (state !== STATE.RUNNING) return;
      drawBoxes(boxes);
      updateStatus(boxes, inference_ms);
    } catch (e) {
      if (e.name === 'AbortError') return;
      state = STATE.ERROR;
      showStatus(`Connection lost: ${e.message}`);
      return;
    }
  }
}

function captureFrame() {
  const targetW = 480;
  const targetH = Math.round((video.videoHeight * targetW) / video.videoWidth);
  captureCanvas.width = targetW;
  captureCanvas.height = targetH;
  captureCtx.drawImage(video, 0, 0, targetW, targetH);
  return new Promise((resolve) => captureCanvas.toBlob(resolve, 'image/jpeg', 0.7));
}

function sizeOverlayToVideo() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

function drawBoxes(boxes) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const b of boxes) {
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    const w = (b.x2 - b.x1) * overlay.width;
    const h = (b.y2 - b.y1) * overlay.height;
    ctx.lineWidth = 3;
    ctx.strokeStyle = b.class === 'smiling' ? '#22c55e' : '#94a3b8';
    ctx.strokeRect(x, y, w, h);
    const label = `${b.class} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '14px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(x, y - 20, tw + 8, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 4, y - 5);
  }
}

function updateStatus(boxes, ms) {
  const smiling = boxes.filter((b) => b.class === 'smiling').length;
  const neutral = boxes.filter((b) => b.class === 'neutral').length;
  statusEl.textContent = `Smiling: ${smiling} · Neutral: ${neutral} · Inference: ${ms.toFixed(0)} ms`;
}

function showStatus(msg) {
  statusEl.textContent = msg;
}

toggleBtn.addEventListener('click', () => {
  if (state === STATE.RUNNING) stop();
  else start();
});

window.addEventListener('resize', () => {
  if (state === STATE.RUNNING) sizeOverlayToVideo();
});
