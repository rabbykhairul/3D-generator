# Deploy Frame Lab to root@78.47.70.115

This guide deploys the Python reconstruction API and its single CPU worker,
built React frontend, SQLite/uploads/artifacts, HTTPS, and password protection.
The supplied `Dockerfile`, `compose.yaml`, and `deploy/Caddyfile` implement it.
No deployment was performed from the coding session: outbound SSH was denied.
Run these commands in your own terminal, where SSH works.

## 1. Prerequisites and safety

- A Linux server with Docker Engine and Compose v2. Commands below assume Ubuntu
  24.04; confirm with `cat /etc/os-release` before installing packages.
- Start with 4 CPU cores, 8 GB RAM and 30 GB free disk as an operational estimate,
  not a benchmark. Large photos and concurrent uploads need more memory/storage.
- A domain/subdomain you control, for example `frames.example.com`. Point its A
  record to `78.47.70.115`. Remove an incorrect AAAA record unless IPv6 also works.
- TCP 80 and 443 reachable for HTTPS; TCP 22 restricted to your administration IP
  where possible. Do not expose port 8018, SQLite, or Docker's socket publicly.
- This is a shared workspace, **not a multi-tenant merchant service**. Everyone
  with its password can access all projects. Do not make it anonymous/public.
- Back up any existing server application before changing proxy configuration.

From your Mac:

```sh
ssh root@78.47.70.115
```

On the server, inspect rather than stop existing services:

```sh
cat /etc/os-release
df -h /
free -h
ss -ltnp
docker --version
docker compose version
```

If Docker is missing, follow the [official Ubuntu installation instructions](https://docs.docker.com/engine/install/ubuntu/)
for the confirmed OS, including the Compose plugin. Avoid uninstalling an
existing Docker installation used by other applications.

**Ports 80/443 already in use?** Do not stop Nginx, Apache, Caddy or another app.
Use a separate hostname/vhost through that existing proxy. See section 8.
Docker-published ports can bypass some UFW rules; enforce exposure restrictions
at the provider firewall too. Never enable a firewall without retaining SSH.

## 2. Upload application source (Mac)

Exit the SSH session first. Upload only source, not the Mac Python environment,
node_modules, generated models, secrets or the live database:

```sh
cd /Users/dekuddus/Herd/salgeni/eyewear-reconstruction-demo
ssh root@78.47.70.115 'mkdir -p /opt/eyewear'
rsync -av --exclude='.git' --exclude='.venv' --exclude='node_modules' \
  --exclude='data' --exclude='artifacts' --exclude='dist' \
  --exclude='.env*' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='test-results' --exclude='playwright-report' \
  ./ root@78.47.70.115:/opt/eyewear/
```

No `--delete` is used. This is a fresh installation with an empty project store;
local merchant projects are deliberately not uploaded implicitly. Public sample
photos from local `data/references` are also excluded; synthetic samples remain.

## 3. Configure domain and login (server)

```sh
ssh root@78.47.70.115
cd /opt/eyewear
docker run --rm -it caddy:2 caddy hash-password
cp -n deploy/env.example .env
chmod 600 .env
nano .env
```

The password command prompts securely. Paste its **hash**, not your password:

```dotenv
APP_DOMAIN=frames.example.com
AUTH_USER=merchant
AUTH_HASH='$2a$14$REPLACE_WITH_THE_COMPLETE_GENERATED_HASH'
RELEASE_TAG=engine34
```

Replace the domain. Use no `https://`, path or trailing slash in `APP_DOMAIN`.
Keep single quotes around the hash so Compose treats its `$` characters
literally. Do not commit or share `.env`.

## 4. Build and start

```sh
cd /opt/eyewear
docker compose config --quiet
docker compose build --pull
docker compose run --rm --no-deps web caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
docker compose up -d
docker compose ps
docker compose logs --tail=100 api web
```

Caddy obtains/renews the domain certificate and redirects HTTP to HTTPS. DNS and
ports must be correct. Allow time for the first certificate issuance. The API
runs as non-root and has no published host port. One Uvicorn process starts one
worker; do not add Uvicorn workers or scale API replicas with this SQLite design.

Data resides in Docker volume `eyewear_eyewear_data` at `/data` inside the API.
TLS state resides in `eyewear_caddy_data` and `eyewear_caddy_config`. Restarting
containers or rebuilding source does not remove these volumes.

## 5. Verify before handing to merchants

Replace the domain below. Curl prompts for the password instead of saving it in
shell history:

```sh
curl -I https://frames.example.com/
curl -u merchant https://frames.example.com/api/health
curl -u merchant -H 'Origin: https://frames.example.com' https://frames.example.com/api/health
curl -u merchant -H 'Origin: https://untrusted.example' https://frames.example.com/api/health
```

Expected: unauthenticated request gets 401, authenticated health gets
`{"status":"ok",...}`, allowed origin gets 200, foreign origin gets 403.
The entire frontend, photos, models and API require proxy authentication.
Origin checks are additional browser protection, not a substitute for login.

Open `https://frames.example.com` and:

1. Log in, create a test project, upload front and side photos.
2. Analyze, review/approve contours, generate, and wait for the job to complete.
3. Download the GLB and reload the page; confirm persistence.
4. Test camera/AR with permission granted. HTTPS is required for remote camera
   use; insecure `http://78.47.70.115` is not the supported deployment.
5. Confirm source comparison and current engine metadata (29).

Face-tracking WASM and models are bundled under `frontend/public/wasm` and
`frontend/public/models`; keep them in the uploaded source. The browser loads
them from this same authenticated site. Reconstruction runs on the server CPU.
Review third-party model/asset licensing before commercial use.

## 6. Backup, updates and rollback

Wait for active generation to finish. Stop API writes while copying SQLite and
its associated files; the web container may temporarily return 502 during this
short maintenance window:

```sh
cd /opt/eyewear
mkdir -p backups
chmod 700 backups
docker compose stop api
docker compose run --rm --no-deps --user 0 -v /opt/eyewear/backups:/backup api \
  sh -c 'tar -czf /backup/data-$(date -u +%Y%m%dT%H%M%SZ).tar.gz -C /data .'
docker compose start api
chmod 600 backups/*.tar.gz
```

Copy backups off the server through SSH to encrypted storage. Separately retain
the source release and `.env` securely. Test restoring into a **new** volume and
temporary installation before trusting a backup. Never extract over a live
database. Preserve Caddy volumes to avoid unnecessary certificate reissuance.

For an update:

1. Back up data and record `docker compose images` and the old `RELEASE_TAG`.
2. Upload changed source using section 2; `.env` and data remain untouched.
3. Set a new, unique `RELEASE_TAG` (e.g. `engine34-20261004`); do not reuse tags.
4. Run `docker compose build`, then `docker compose up -d` and section 5 checks.
5. Reanalyze/approve/regenerate old projects when geometry changes require it.

If an update fails and the database schema is compatible, restore the previous
release's compose/config files and set its old tag, then:

```sh
docker compose up -d --no-build --pull never
```

Keep previous images until the new release is accepted. If a schema migration
occurred, restore the matching data backup into a separate volume first; merely
rolling back images is insufficient. Do not run `docker compose down -v` or volume
pruning: those operations destroy persisted data.

### Moving existing local projects

Do not blindly copy `studio.sqlite3`: it stores absolute Mac photo paths, and job
snapshots/artifact records also need handling. For the initial deployment,
reupload photos into fresh projects. A full migration requires a stopped-source
backup of the complete `data/` directory, rebasing stored paths to `/data`, and
verifying every photo/artifact in an isolated target before switching. Do not
merge it into a populated server database. Migration is not automated by this
deployment package.

## 7. Troubleshooting

```sh
cd /opt/eyewear
docker compose ps
docker compose logs --tail=200 api
docker compose logs --tail=200 web
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8018/api/health').read().decode())"
docker stats --no-stream
df -h /
```

- **502:** API unhealthy/crashed; inspect API logs and memory/disk availability.
- **403 on uploads:** `APP_DOMAIN`/Origin mismatch; use the exact HTTPS hostname.
  Do not disable origin checking globally.
- **401:** missing/wrong credentials or malformed bcrypt hash. Recreate the web
  container after editing `.env`: `docker compose up -d --force-recreate web`.
- **TLS fails:** check DNS A/AAAA, 80/443 firewall and proxy conflicts. Avoid
  repeated certificate requests while DNS is wrong.
- **Upload too large:** API limit is 20 MiB per image and 40 million pixels;
  proxy limit is 25 MB including multipart overhead.
- **Jobs queued:** check worker startup/lock and available memory. Do not start
  duplicate workers. Interrupted jobs need regeneration after restart.
- **Old appearance:** regenerate older exports with the current engine; existing GLBs
  are immutable and are not silently replaced.
- **SSH works locally but not in Codex:** deploy from your terminal; the coding
  session's restricted network permissions are independent of server credentials.

## 8. Existing reverse proxy owns ports 80/443

Use that proxy instead of publishing a second TLS server. A safe integration is:

1. Change this compose web binding to `127.0.0.1:8088:80` only, removing 443.
2. Change the first line of `deploy/Caddyfile` to `:80` (internal HTTP). Keep its
   authentication and API/frontend routing unchanged.
3. Keep `APP_DOMAIN` equal to the external HTTPS hostname so API origin checks
   continue to work.
4. Add a new HTTPS vhost at the existing proxy forwarding **all paths** to
   `http://127.0.0.1:8088`. Preserve `Host` and `Authorization`; allow 25 MB bodies.
5. Validate the existing proxy config, reload rather than stop it, and perform
   section 5's authentication/origin/upload tests again.

Inspect the actual server configuration before making this adaptation. Do not
replace another application's vhost, default site or certificates.

## References

- [Docker Ubuntu installation](https://docs.docker.com/engine/install/ubuntu/)
- [Compose environment quoting](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)
- [Caddy password authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth)
- [Caddy reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
- [Caddy HTTPS and configuration](https://caddyserver.com/docs/caddyfile-tutorial)

These files were prepared locally. A Docker image build, TLS issuance, remote
restart and live-browser acceptance must still be verified on the target server.

SAM 3 CPU is an optional isolated runtime; follow the README SAM 3 setup.
It requires approved official checkpoint access and is not part of the existing
SAM 2 container image. Set `EYEWEAR_SAM3_PYTHON` and `EYEWEAR_SAM3_MANIFEST`
to installed paths if different from the local defaults. Do not select SAM 3
for production until real inference latency, memory and eyewear accuracy have
been evaluated. Local CPU inference now passes on a saved front photo. The general `lens` prompt
produced a usable optical pair, while connected frame/bridge geometry remained
unresolved. This is not a production accuracy validation.
