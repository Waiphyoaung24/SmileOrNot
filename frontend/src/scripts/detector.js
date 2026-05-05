// Parameterized detector controller. Used by the smile and can pages.
//
// Usage (live mode):
//   initLiveDetector({
//     endpoint: '/predict',
//     classColors: { smiling: '#22c55e', neutral: '#94a3b8' },
//     elements: { video, overlay, status, toggle },
//   });
//
// Usage (still mode):
//   detectStill({ endpoint: '/predict/can', classColors, file, overlay, image, status });

const STATE = { IDLE: 'idle', RUNNING: 'running', ERROR: 'error' };

export function initLiveDetector({ endpoint, classColors, elements }) {
  const { video, overlay, status, toggle } = elements;
  const ctx = overlay.getContext('2d');
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');
  let state = STATE.IDLE;
  let stream = null;
  let abortController = null;

  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, facingMode: { ideal: 'environment' } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      sizeOverlay();
      state = STATE.RUNNING;
      toggle.textContent = 'Stop Camera';
      loop();
    } catch (e) {
      state = STATE.ERROR;
      status.textContent = e.name === 'NotAllowedError' ? 'Camera permission denied.' : 'Camera unavailable.';
    }
  }

  function stop() {
    state = STATE.IDLE;
    abortController?.abort();
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    toggle.textContent = 'Start Camera';
    status.textContent = 'Stopped.';
  }

  async function loop() {
    while (state === STATE.RUNNING) {
      try {
        const blob = await captureFrame();
        abortController = new AbortController();
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const r = await fetch(endpoint, { method: 'POST', body: fd, signal: abortController.signal });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { boxes, inference_ms } = await r.json();
        if (state !== STATE.RUNNING) return;
        drawBoxes(ctx, overlay, boxes, classColors);
        renderStatus(status, boxes, inference_ms);
      } catch (e) {
        if (e.name === 'AbortError') return;
        state = STATE.ERROR;
        status.textContent = `Connection lost: ${e.message}`;
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

  function sizeOverlay() {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }

  toggle.addEventListener('click', () => (state === STATE.RUNNING ? stop() : start()));
  window.addEventListener('resize', () => state === STATE.RUNNING && sizeOverlay());
}

export async function detectStill({ endpoint, classColors, file, overlay, image, status }) {
  const ctx = overlay.getContext('2d');
  const url = URL.createObjectURL(file);
  await new Promise((res) => {
    image.onload = res;
    image.src = url;
  });
  overlay.width = image.clientWidth;
  overlay.height = image.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  status.textContent = 'Predicting…';
  const fd = new FormData();
  fd.append('file', file, file.name);
  const r = await fetch(endpoint, { method: 'POST', body: fd });
  if (!r.ok) {
    status.textContent = `Error: HTTP ${r.status}`;
    return;
  }
  const { boxes, inference_ms } = await r.json();
  drawBoxes(ctx, overlay, boxes, classColors);
  renderStatus(status, boxes, inference_ms);
}

function drawBoxes(ctx, overlay, boxes, classColors) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const b of boxes) {
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    const w = (b.x2 - b.x1) * overlay.width;
    const h = (b.y2 - b.y1) * overlay.height;
    const color = classColors[b.class] ?? '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const label = `${b.class} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '14px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 20, tw + 8, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x + 4, y - 5);
  }
}

function renderStatus(statusEl, boxes, ms) {
  const counts = boxes.reduce((acc, b) => ((acc[b.class] = (acc[b.class] || 0) + 1), acc), {});
  const parts = Object.entries(counts).map(([k, v]) => `${k}: ${v}`);
  statusEl.textContent = `${parts.join(' · ') || 'no detections'} · ${ms.toFixed(0)} ms`;
}
