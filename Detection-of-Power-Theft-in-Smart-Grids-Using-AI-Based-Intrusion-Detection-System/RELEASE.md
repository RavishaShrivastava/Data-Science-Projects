# Release Checklist (Demo)

## Pre-flight
- [ ] `pytest -q` passes
- [ ] `ruff check .` passes
- [ ] `pip install -r requirements.txt` succeeds

## Configuration
- [ ] Set `DATA_MODE=simulated` for demo consistency (default)
- [ ] If `DATA_MODE=real`, set `DATASET_PATH` and verify dataset loads
- [ ] If exposing `/api/*` publicly, set `API_KEY`
- [ ] If `FLASK_ENV=production`, set `SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`

## Run locally
- [ ] `make run`
- [ ] Verify `/health` returns `{ "status": "ok" }`
- [ ] Verify `/api/available-years` works with `X-API-KEY` when configured

## Docker
- [ ] `make docker-build`
- [ ] `make docker-run`

## Deployment (Railway/Heroku-style)
- [ ] Ensure `PORT` is provided by platform
- [ ] Ensure `gunicorn` starts successfully (Procfile / railway.json)
