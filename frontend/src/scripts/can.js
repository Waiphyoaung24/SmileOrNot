import { initLiveDetector, detectStill } from './detector.js';

const CAN_COLORS = { can: '#94fcff' };

// The training set had only can-containing images, so the model fits a "can"
// box to anything when the scene is empty. Drop low-confidence and
// implausibly-large detections before rendering.
const MIN_CONF = 0.6;
const MAX_AREA = 0.5;       // > 50% of frame → almost certainly bogus
const MAX_SIDE = 0.85;      // a single side > 85% of frame → bogus
const filter = (boxes) =>
  boxes.filter((b) => {
    if (b.conf < MIN_CONF) return false;
    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    if (w * h > MAX_AREA) return false;
    if (w > MAX_SIDE || h > MAX_SIDE) return false;
    return true;
  });

const status = document.getElementById('status');
const livePane = document.getElementById('live-pane');
const uploadPane = document.getElementById('upload-pane');
const liveBtn = document.getElementById('mode-live');
const uploadBtn = document.getElementById('mode-upload');
const startBtn = document.getElementById('toggle');
const uploadLabel = document.getElementById('upload-btn');

initLiveDetector({
  endpoint: '/predict/can',
  classColors: CAN_COLORS,
  filter,
  facingMode: 'environment',
  elements: {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay-live'),
    status,
    toggle: startBtn,
  },
});

const fileInput = document.getElementById('file-input');
const image = document.getElementById('upload-image');
const overlayUpload = document.getElementById('overlay-upload');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  await detectStill({
    endpoint: '/predict/can',
    classColors: CAN_COLORS,
    filter,
    file,
    overlay: overlayUpload,
    image,
    status,
  });
});

function setMode(mode) {
  const live = mode === 'live';
  livePane.hidden = !live;
  uploadPane.hidden = live;
  liveBtn.setAttribute('aria-selected', String(live));
  uploadBtn.setAttribute('aria-selected', String(!live));
  startBtn.hidden = !live;
  uploadLabel.hidden = live;
}

liveBtn.addEventListener('click', () => setMode('live'));
uploadBtn.addEventListener('click', () => setMode('upload'));
