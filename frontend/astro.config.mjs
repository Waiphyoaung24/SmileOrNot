// @ts-check
import { defineConfig } from 'astro/config';
import http from 'node:http';

const BACKEND = 'http://localhost:8000';

// Astro's static dev middleware returns 501 on POST before Vite's proxy runs.
// Inject our own middleware via astro:server:setup so it runs first.
function backendProxy() {
  return {
    name: 'backend-proxy',
    hooks: {
      'astro:server:setup': (/** @type {{ server: import('vite').ViteDevServer }} */ { server }) => {
        server.middlewares.use((/** @type {import('http').IncomingMessage} */ req, /** @type {import('http').ServerResponse} */ res, /** @type {(err?: any) => void} */ next) => {
          if (!req.url?.startsWith('/predict')) return next();
          const url = new URL(BACKEND + req.url);
          const proxyReq = http.request(
            {
              hostname: url.hostname,
              port: url.port,
              path: url.pathname + url.search,
              method: req.method,
              headers: { ...req.headers, host: url.host },
            },
            (proxyRes) => {
              res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
              proxyRes.pipe(res);
            },
          );
          proxyReq.on('error', (err) => {
            res.statusCode = 502;
            res.end(`Backend unreachable: ${err.message}`);
          });
          req.pipe(proxyReq);
        });
      },
    },
  };
}

export default defineConfig({
  output: 'static',
  outDir: '../smileornot/static',
  integrations: [backendProxy()],
});
