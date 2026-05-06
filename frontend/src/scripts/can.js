import { initLiveDetector, detectStill } from './detector.js';

const CAN_COLORS = { can: '#7cff8d' };

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
