# WaveSplit Cloudflare Tunnel Setup

WaveSplit listens locally on `127.0.0.1:8001`. Cloudflare Tunnel should publish that local origin as an HTTPS hostname.

## Local App

Build the frontend and start the API:

```bash
cd /wkspace/wavesplit
cd web && npm run build && cd ..
WAVESPLIT_AUTH_COOKIE_SECURE=true uvicorn wavesplit.api:app --host 127.0.0.1 --port 8001
```

Sign-in credentials are loaded from `config.yaml`:

```yaml
auth:
  users:
    - username: admin
      password: change-me-now
```

Change the default password and `auth.session_secret` before exposing the app.

## Cloudflare Dashboard Setup

1. In Cloudflare, make sure the domain you want to use is added to your account and is using Cloudflare nameservers.
2. Open **Zero Trust**.
3. Go to **Networks** -> **Tunnels**.
4. Select **Create a tunnel**.
5. Choose **Cloudflared**.
6. Name the tunnel, for example `wavesplit`.
7. Choose your operating system and copy the install command Cloudflare shows.
8. Copy only the tunnel token from the command Cloudflare shows. Store it in `/etc/wavesplit/cloudflared.token`; do not put it in a systemd environment file because `cloudflared` logs environment variables at startup.
9. Add a **Public Hostname**:
   - Subdomain: `wavesplit`
   - Domain: your Cloudflare-managed domain
   - Type: `HTTP`
   - URL: `127.0.0.1:8001`
10. Save the tunnel and open `https://wavesplit.your-domain.example`.

## Token-Based Service Install

After installing `cloudflared`, create the token file:

```bash
sudo mkdir -p /etc/wavesplit
sudo editor /etc/wavesplit/cloudflared.token
sudo chmod 600 /etc/wavesplit/cloudflared.token
```

Install the app service examples:

```bash
sudo cp deploy/cloudflare/wavesplit.service.example /etc/systemd/system/wavesplit.service
sudo cp deploy/cloudflare/wavesplit-worker.service.example /etc/systemd/system/wavesplit-worker.service
sudo cp deploy/cloudflare/cloudflared-token.service.example /etc/systemd/system/cloudflared-wavesplit.service
sudo systemctl daemon-reload
sudo systemctl enable --now redis-server.service wavesplit.service wavesplit-worker.service
sudo systemctl enable --now cloudflared-wavesplit.service
```

Only start `cloudflared-wavesplit.service` after `/etc/wavesplit/cloudflared.token` contains a real Cloudflare tunnel token.

Check status:

```bash
systemctl status redis-server.service wavesplit.service wavesplit-worker.service cloudflared-wavesplit.service
journalctl -u wavesplit.service -u wavesplit-worker.service -u cloudflared-wavesplit.service -f
```

## Local Config File Alternative

If you prefer a locally managed named tunnel instead of a dashboard token, use `deploy/cloudflare/config.yml.example` as a template:

```bash
cloudflared tunnel login
cloudflared tunnel create wavesplit
cloudflared tunnel route dns wavesplit wavesplit.your-domain.example
cp deploy/cloudflare/config.yml.example ~/.cloudflared/config.yml
editor ~/.cloudflared/config.yml
cloudflared tunnel run wavesplit
```

## Security Notes

- The app protects `/api/*` with the local username/password session cookie.
- Static frontend assets are public so the sign-in page can load; data, uploads, downloads, clip previews, reports, and health checks require login.
- Keep `auth.cookie_secure: false` for plain local HTTP. Use `WAVESPLIT_AUTH_COOKIE_SECURE=true` or set `auth.cookie_secure: true` when accessing through Cloudflare HTTPS.
- Consider adding Cloudflare Access in front of the tunnel for an additional identity layer.
