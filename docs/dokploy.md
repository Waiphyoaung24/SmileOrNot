# Deploying SmileOrNot on Dokploy

## Files

- `Dockerfile` — multi-stage build (Node frontend → Python runtime). EXPOSE 7860.
- `docker-compose.yml` — Compose stack with Traefik labels + `dokploy-network`.

## Prerequisites

- Dokploy instance reachable, with Traefik + Let's Encrypt configured.
- A domain or subdomain (e.g. `smileornot.example.com`) with DNS A record pointing to your Dokploy host.
- This repo accessible (Git URL or upload).

## Deploy via Dokploy UI

1. **Project → + Create Service → Compose.**
2. **Source:** point at your Git repo (branch `main`).
3. **Compose Path:** `docker-compose.yml`.
4. **Edit Compose** in the Dokploy editor: replace `smileornot.your-domain.com` with your real hostname in the `Host(...)` Traefik label. Save.
5. **Deploy.** First build is slow (~3–5 min: pulls torch CPU wheels + ultralytics + node deps).
6. Watch logs in the Deployments tab. When uvicorn prints `Application startup complete.`, hit your domain.

## Verifying

```bash
curl https://smileornot.example.com/                       # 200, HTML
curl -X POST -F "file=@some-face.jpg" \
     https://smileornot.example.com/predict                # JSON {boxes, inference_ms}
```

In the browser: open the page, allow camera, click Start, see live boxes.

## Updating

Push to `main` → Dokploy auto-deploys (if the project's auto-deploy is on). Otherwise click **Redeploy** in the Compose service page.

## Notes

- No env vars needed — model path and class names are baked into the image.
- The container runs as non-root `user` (uid 1000).
- Healthcheck pings `GET /` every 30s; Traefik will mark the service down if it fails.
- If you want the API on its own subdomain (e.g. `api.smileornot.example.com`), add a second router:
  ```
  - traefik.http.routers.smileornot-api.rule=Host(`api.smileornot.example.com`)
  - traefik.http.routers.smileornot-api.entrypoints=websecure
  - traefik.http.routers.smileornot-api.tls.certResolver=letsencrypt
  ```
  Both routers point at the same `loadbalancer.server.port=7860`.
