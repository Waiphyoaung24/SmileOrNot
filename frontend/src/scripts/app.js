import { initLiveDetector } from './detector.js';

initLiveDetector({
  endpoint: '/predict',
  classColors: { smiling: '#94fcff', neutral: '#b9afbb' },
  facingMode: 'user',
  elements: {
    video: document.getElementById('video'),
    overlay: document.getElementById('overlay'),
    status: document.getElementById('status'),
    toggle: document.getElementById('toggle'),
    flip: document.getElementById('flip'),
  },
});
