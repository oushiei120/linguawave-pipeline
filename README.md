# linguawave-pipeline

Public cloud transcription pipeline for [LinguaWave 语浪](https://github.com/oushiei120/linguawave) (App stays in a private repo).

Daily KOTO worker: RSS → CBR audio on Tencent COS → Groq Whisper → DeepSeek translation → GiNZA bunsetsu → publish gate → `index.json`.

## GitHub Actions secrets

| Secret | Required |
|--------|----------|
| `GROQ_API_KEY` | yes (`cloud_transcribe`) |
| `DEEPSEEK_API_KEY` | yes |
| `TENCENT_COS_SECRET_ID` | yes |
| `TENCENT_COS_SECRET_KEY` | yes |
| `TENCENT_COS_REGION` | optional (default `ap-shanghai`) |
| `TENCENT_COS_BUCKET` | optional (default `koto-audio-1314383219`) |

## Workflows

- **Cloud Audio Transcribe** — cron `0 20 * * *` (04:00 Beijing) + `workflow_dispatch`
- **Rebuild Published Library** — `workflow_dispatch` only

Smoke test: Actions → Cloud Audio Transcribe → Run workflow with `max_episodes` = `1` or `3`.

## Local ops

```bash
# blocked-word list lives on COS (private), never in git
python scripts/policy_tool.py pull   # → ./content_policy.json (gitignored)
# edit content_policy.json
python scripts/policy_tool.py push
```

Needs COS credentials in `.env` (also gitignored).

## Layout

Keep paths as-is (`scripts/`, root `bunsetsu.py`, `source_catalog.json`). Flattening breaks `REPO_ROOT` and workflow self-checks.
