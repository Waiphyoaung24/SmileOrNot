// Parameterized detector controller. Used by both the smile and can pages.
//
// Renders into the Vision-Lab HUD when present:
//   [data-stat="count"]   → integer detection count
//   [data-stat="ms"]      → inference time (number only, "ms" unit is sibling)
//   [data-stat="classes"] → human breakdown ("smiling: 1 · neutral: 2")
//   [data-rec]            → REC chip; toggles .is-rec
//   [data-rec-label]      → "Standby" | "Live" | "Stopped" | "Error"
//   [data-viewport]       → adds .is-running while streaming for scanline/grid

const STATE = { IDLE: 'idle', RUNNING: 'running', ERROR: 'error' };

export function initLiveDetector({ endpoint, classColors, elements }) {
  const { video, overlay, status, toggle } = elements;
  const ctx = overlay.getContext('2d');
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');
  let state = STATE.IDLE;
  let stream = null;
  let abortController = null;

  const viewport = video.closest('[data-viewport]');
  const recChip = document.querySelector('[data-rec]');
  const recLabel = document.querySelector('[data-rec-label]');

  setRec('Standby', false);

  async function start() {
    try {
      setStatus('Requesting camera…');
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, facingMode: { ideal: 'environment' } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      sizeOverlay();
      state = STATE.RUNNING;
      viewport?.classList.add('is-running');
      toggle.textContent = 'Stop · Camera';
      toggle.dataset.state = 'running';
      setRec('Live', true);
      setStatus('Streaming.');
      loop();
    } catch (e) {
      state = STATE.ERROR;
      const msg =
        e.name === 'NotAllowedError'
          ? 'Camera permission denied.'
          : 'Camera unavailable.';
      setRec('Error', false);
      setStatus(msg);
    }
  }

  function stop() {
    state = STATE.IDLE;
    abortController?.abort();
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    viewport?.classList.remove('is-running');
    toggle.textContent = 'Start · Camera';
    delete toggle.dataset.state;
    setRec('Stopped', false);
    setStatus('Idle.');
    resetStats();
  }

  async function loop() {
    while (state === STATE.RUNNING) {
      try {
        const blob = await captureFrame();
        abortController = new AbortController();
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const r = await fetch(endpoint, {
          method: 'POST',
          body: fd,
          signal: abortController.signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { boxes, inference_ms } = await r.json();
        if (state !== STATE.RUNNING) return;
        drawBoxes(ctx, overlay, boxes, classColors);
        updateStats(boxes, inference_ms);
      } catch (e) {
        if (e.name === 'AbortError') return;
        state = STATE.ERROR;
        viewport?.classList.remove('is-running');
        setRec('Error', false);
        setStatus(`Connection lost: ${e.message}`);
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
    return new Promise((resolve) =>
      captureCanvas.toBlob(resolve, 'image/jpeg', 0.7),
    );
  }

  function sizeOverlay() {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  }

  function setStatus(text) {
    if (status) status.textContent = text;
  }

  function setRec(label, on) {
    if (recLabel) recLabel.textContent = label;
    if (recChip) recChip.classList.toggle('is-rec', !!on);
  }

  toggle.addEventListener('click', () =>
    state === STATE.RUNNING ? stop() : start(),
  );
  window.addEventListener('resize', () => state === STATE.RUNNING && sizeOverlay());
}

export async function detectStill({
  endpoint,
  classColors,
  file,
  overlay,
  image,
  status,
}) {
  const ctx = overlay.getContext('2d');
  const url = URL.createObjectURL(file);
  await new Promise((res) => {
    image.onload = res;
    image.src = url;
  });
  overlay.width = image.clientWidth;
  overlay.height = image.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (status) status.textContent = 'Predicting…';
  const fd = new FormData();
  fd.append('file', file, file.name);
  const r = await fetch(endpoint, { method: 'POST', body: fd });
  if (!r.ok) {
    if (status) status.textContent = `Error: HTTP ${r.status}`;
    return;
  }
  const { boxes, inference_ms } = await r.json();
  drawBoxes(ctx, overlay, boxes, classColors);
  updateStats(boxes, inference_ms);
  if (status) status.textContent = 'Done.';
}

function drawBoxes(ctx, overlay, boxes, classColors) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const b of boxes) {
    const x = b.x1 * overlay.width;
    const y = b.y1 * overlay.height;
    const w = (b.x2 - b.x1) * overlay.width;
    const h = (b.y2 - b.y1) * overlay.height;
    const color = classColors[b.class] ?? '#7cff8d';
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    // Corner brackets on each box for the instrument feel.
    const t = Math.min(14, w * 0.2, h * 0.2);
    ctx.beginPath();
    ctx.moveTo(x, y + t); ctx.lineTo(x, y); ctx.lineTo(x + t, y);
    ctx.moveTo(x + w - t, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + t);
    ctx.moveTo(x, y + h - t); ctx.lineTo(x, y + h); ctx.lineTo(x + t, y + h);
    ctx.moveTo(x + w - t, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - t);
    ctx.lineWidth = 3;
    ctx.stroke();

    // Label tag.
    const label = `${b.class.toUpperCase()} ${(b.conf * 100).toFixed(0)}%`;
    ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 18, tw + 10, 18);
    ctx.fillStyle = '#07090b';
    ctx.fillText(label, x + 5, y - 5);
  }
}

let lastCount = -1;

function updateStats(boxes, ms) {
  const countEl = document.querySelector('[data-stat="count"]');
  const msEl = document.querySelector('[data-stat="ms"]');
  const classesEl = document.querySelector('[data-stat="classes"]');

  const count = boxes.length;
  if (countEl) {
    countEl.textContent = String(count).padStart(2, '0');
    if (count !== lastCount) {
      const chip = countEl.closest('.chip');
      if (chip) {
        chip.classList.remove('is-pulse');
        // force reflow so animation restarts.
        void chip.offsetWidth;
        chip.classList.add('is-pulse');
      }
      lastCount = count;
    }
  }
  if (msEl) msEl.textContent = ms.toFixed(0);
  if (classesEl) {
    if (count === 0) {
      classesEl.textContent = 'none';
    } else {
      const counts = boxes.reduce(
        (acc, b) => ((acc[b.class] = (acc[b.class] || 0) + 1), acc),
        {},
      );
      classesEl.textContent = Object.entries(counts)
        .map(([k, v]) => `${k}: ${v}`)
        .join(' · ');
    }
  }
}

function resetStats() {
  lastCount = -1;
  const countEl = document.querySelector('[data-stat="count"]');
  const msEl = document.querySelector('[data-stat="ms"]');
  if (countEl) countEl.textContent = '—';
  if (msEl) msEl.textContent = '—';
}
