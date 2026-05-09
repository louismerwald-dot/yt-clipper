# yt-clipper

Daily, hands-off pipeline that watches a list of YouTube channels, clips the latest video into vertical Shorts with burned-in captions, and auto-uploads to your channel. Runs on GitHub Actions for free.

## What it does, every morning at 08:30 UTC

1. For each source channel in `config.yaml`, fetches the latest video
2. Skips ones already processed (tracked in `state.json`)
3. Downloads with yt-dlp, transcribes with `faster-whisper`
4. Sends the transcript to **Gemini 2.0 Flash** (free tier) and asks for the best ~3 viral moments
5. Cuts each moment with FFmpeg, crops 9:16, burns big bouncy captions
6. Uploads to your YouTube channel as a Short

## Honest expectations

- **This won't make a dead channel monetized.** It's a content engine. Picking the right niche + sources matters more than the code.
- **Copyright risk is real.** Re-uploading other creators' content can trigger Content ID claims, copyright strikes, channel termination, or demonetization. Best practices to lower risk: pick creators known to allow clips, add commentary/edits, credit the source in the description, and don't clip music-heavy content. **You're responsible for what you upload.**
- **Free tiers have limits.** YouTube Data API: ~6 uploads/day on default quota. Gemini free tier: rate-limited but plenty for daily transcripts. GitHub Actions: 2,000 free minutes/month for private repos, **unlimited for public repos**. Make this repo public and it's effectively free forever.
- **YouTube actively breaks yt-dlp.** The workflow auto-installs the freshest yt-dlp and Deno (now required). If a run fails, usually waiting a day for a yt-dlp release fixes it. The `YT_COOKIES` secret (optional) helps a lot — see below.

## One-time setup (~20 minutes)

### 1. Fork or push this repo to your GitHub

Make it **public** if you want unlimited Actions minutes (recommended).

### 2. Get a Gemini API key (free)

- Go to https://aistudio.google.com/apikey
- Create an API key. Save it.

### 3. Set up a Google Cloud project for the YouTube Data API

- https://console.cloud.google.com/ — create a new project
- "APIs & Services" → "Library" → enable **YouTube Data API v3**
- "APIs & Services" → "OAuth consent screen" → External, add your email as a test user, scopes can be empty for now
- "APIs & Services" → "Credentials" → "Create Credentials" → **OAuth client ID** → **Desktop app**
- Download the `client_secret.json`

### 4. Generate your YouTube refresh token (one-time, on your laptop)

```bash
pip install google-auth-oauthlib
python get_refresh_token.py /path/to/client_secret.json
```

A browser will pop up — log in with the Google account that owns your YouTube channel, grant access. The script prints `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`. Keep these.

### 5. (Optional but recommended) Export YouTube cookies

YouTube increasingly blocks unauthenticated yt-dlp. To use cookies:
- Install a browser extension like "Get cookies.txt LOCALLY"
- Visit youtube.com while logged into a **throwaway Google account** (not your main one — there's a small risk of YouTube flagging it)
- Export cookies for youtube.com in **Netscape format**
- Open the file, copy its entire contents

### 6. Add secrets to your GitHub repo

`Settings → Secrets and variables → Actions → New repository secret`. Add:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | from step 2 |
| `YT_CLIENT_ID` | from step 4 |
| `YT_CLIENT_SECRET` | from step 4 |
| `YT_REFRESH_TOKEN` | from step 4 |
| `YT_COOKIES` | (optional) full contents of cookies.txt from step 5 |

### 7. Edit `config.yaml`

- Replace the `channels:` URLs with 3–5 sources in your niche
- Tweak hashtags, clip lengths, daily upload cap

### 8. Test it manually

`Actions` tab → `daily-clipper` → `Run workflow`. Watch the logs. First run usually takes 5–15 minutes depending on source video length.

After that it runs daily on its own.

## Tuning for monetization speed

- **Pick ONE niche.** AI news, finance, fitness, comedy, sports takes — pick one and stay there. Algorithm needs consistency.
- **Pick the right sources.** Long-form podcasts are gold (lots of moments per video, conversational hooks). 1–3 hour interview shows are ideal.
- **Title hooks matter most.** The Gemini prompt is tuned for hook-style titles, but you can edit `HIGHLIGHT_PROMPT` in `clipper.py` to lean harder into your style.
- **YouTube Shorts monetization threshold (as of late 2025):** 1,000 subs + either 4,000 watch hours OR 10M Shorts views in 90 days. Shorts is the faster path.
- **Don't spam.** 3–5 quality clips/day from 3 channels beats 20 mediocre ones. The daily cap in `config.yaml` keeps you safe.

## Tuning quality

- Bump `WhisperModel("base", ...)` to `"small"` in `clipper.py` for better captions (slower, still free).
- Edit the caption style block in `build_caption_file` — the colors, font, margins are all in the ASS header.
- Edit `HIGHLIGHT_PROMPT` to bias toward your style (e.g., "prefer moments where the host laughs or sounds shocked").

## Troubleshooting

- **`Sign in to confirm you're not a bot`** from yt-dlp → set `YT_COOKIES` secret.
- **Gemini rate-limited** → free tier is generous but daily; if hit, the run fails — re-run tomorrow or get a paid key.
- **Quota exceeded on upload** → script auto-stops for the day. Resume tomorrow.
- **Run is slow** → reduce `clips_per_video`, or lower max video length in config.

## Files

```
.
├── .github/workflows/daily.yml   # the scheduler
├── clipper.py                    # main pipeline
├── config.yaml                   # your channels + preferences
├── get_refresh_token.py          # run once locally
├── requirements.txt
├── state.json                    # auto-managed: processed video ids
└── README.md
```
