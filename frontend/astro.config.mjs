// @ts-check
import { defineConfig } from 'astro/config';

export default defineConfig({
  output: 'static',
  outDir: '../smileornot/static',
  vite: {
    server: {
      proxy: {
        '/predict': 'http://localhost:8000',
      },
    },
  },
});
