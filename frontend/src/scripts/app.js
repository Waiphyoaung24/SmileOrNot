import { initLiveDetector } from './detector.js';

initLiveDetector({
  endpoint: '/predict',
  classColors: { smiling: '#22c55e', neutral: '#94a3b8' },
  elements: {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay'),
    status: document.getElementById('status'),
    toggle: document.getElementById('toggle'),
  },
});
