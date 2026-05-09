"""
yt-clipper: Auto-clip the latest video from configured channels into vertical
shorts with burned-in captions, then upload to your YouTube channel.

Pipeline:
  1. For each source channel: fetch the latest video (skip if already processed)
  2. Download with yt-dlp
  3. Transcribe with faster-whisper (segments + word timestamps)
  4. Ask Gemini 2.0 Flash to pick the best ~3 viral moments (with hook + reason)
  5. Cut each moment with FFmpeg, crop to 9:16, burn in styled captions
  6. Upload to YouTube as a Short (#Shorts in title/description)
  7. Persist processed-video state so we never double-process

Environment variables required:
  GEMINI_API_KEY      Google AI Studio key (free tier)
  YT_CLIENT_ID        OAuth client id for YouTube Data API
  YT_CLIENT_SECRET    OAuth client secret
  YT_REFRESH_TOKEN    OAuth refresh token (one-time generated locally)
  YT_COOKIES          (optional) Netscape-format cookies file contents for yt-dlp
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
import yt_dlp
from faster_whisper import WhisperModel
from google import genai
from google.genai import types as genai_types
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).parent.resolve()
WORK = ROOT / "work"
STATE_FILE = ROOT / "state.json"
CONFIG_FILE = ROOT / "config.yaml"

# ---------- config + state ----------

def load_config() -> dict:
    with CONFIG_FILE.open() as f:
        return yaml.safe_load(f)

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_video_ids": []}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ---------- yt-dlp ----------

def _ydl_opts(extra: dict | None = None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        # android_vr + tv_simply are the two clients least likely to need PoTokens.
        # We add web as a fallback. Order matters — yt-dlp tries them in sequence.
        "extractor_args": {
            "youtube": {
                "player_client": ["android_vr", "tv_simply", "web_safari"],
                "player_skip": ["webpage", "configs"],
            }
        },
        # Pretend to be a recent mobile/TV user agent — helps with bot checks.
        "http_headers": {
            "User-Agent": (
                "com.google.android.youtube/19.50.40 (Linux; U; Android 14) "
                "gzip"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        # Retry strategies
        "retries": 5,
        "fragment_retries": 5,
        "retry_sleep_functions": {"http": lambda n: min(2 ** n, 30)},
        # Sleep between requests to look more human
        "sleep_interval": 2,
        "max_sleep_interval": 5,
    }
    cookies = (os.environ.get("YT_COOKIES") or "").strip()
    if cookies:
        cookies_path = WORK / "cookies.txt"
        cookies_path.write_text(cookies)
        opts["cookiefile"] = str(cookies_path)
    if extra:
        # Merge extractor_args properly so caller-passed args don't get clobbered
        if "extractor_args" in extra:
            merged = dict(opts["extractor_args"])
            merged.update(extra["extractor_args"])
            extra = {**extra, "extractor_args": merged}
        opts.update(extra)
    return opts

def latest_video_for_channel(channel_url: str) -> dict | None:
    """Return {'id', 'title', 'url', 'duration'} for the most recent upload."""
    feed_url = channel_url.rstrip("/")
    if not feed_url.endswith("/videos"):
        feed_url += "/videos"
    opts = _ydl_opts({"playlistend": 1, "extract_flat": True})
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(feed_url, download=False)
    entries = info.get("entries") or []
    if not entries:
        return None
    e = entries[0]
    return {
        "id": e["id"],
        "title": e.get("title", ""),
        "url": f"https://www.youtube.com/watch?v={e['id']}",
        "duration": e.get("duration") or 0,
    }

def download_video(url: str, out_dir: Path) -> Path:
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")
    opts = _ydl_opts({
        "outtmpl": out_tmpl,
        # cap quality so the runner disk + time stay reasonable
        "format": "bv*[height<=720][vcodec^=avc1]+ba/b[height<=720]/bv*+ba/b",
        "merge_output_format": "mp4",
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return Path(ydl.prepare_filename(info)).with_suffix(".mp4")

# ---------- transcription ----------

@dataclass
class Word:
    start: float
    end: float
    text: str

@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]

def transcribe(video_path: Path) -> list[Segment]:
    # 'base' model: fast on CPU, good enough for highlight selection + captions.
    # Bump to 'small' if you want better quality (~3x slower).
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    out: list[Segment] = []
    for seg in segments:
        words = [Word(w.start, w.end, w.word) for w in (seg.words or [])]
        out.append(Segment(seg.start, seg.end, seg.text.strip(), words))
    return out

# ---------- highlight selection ----------

HIGHLIGHT_PROMPT = """You pick viral short-form clips from podcast/video transcripts.

Goal: identify {n} self-contained moments that would perform on YouTube Shorts / TikTok / Reels.

Strong viral moments have:
  - A clear hook in the first 2 seconds (a surprising claim, question, or reaction)
  - One complete thought (no mid-sentence cut at start or end)
  - Emotional charge: surprising, controversial, funny, insightful, or relatable
  - Self-contained: viewer needs no prior context

Constraints:
  - Length: between {min_len} and {max_len} seconds
  - Start and end on natural sentence boundaries from the transcript timestamps
  - Do not overlap other picks

Return STRICT JSON (no prose, no markdown fences) of shape:
{{
  "clips": [
    {{
      "start": <float seconds>,
      "end": <float seconds>,
      "title": "<<=60 char punchy hook for the YouTube Shorts title>",
      "reason": "<one sentence on why this will perform>"
    }}
  ]
}}

Source video title: {video_title}

Transcript (start_seconds | text):
{transcript}
"""

def pick_highlights(
    segments: list[Segment],
    video_title: str,
    n: int,
    min_len: int,
    max_len: int,
) -> list[dict]:
    transcript_lines = [f"{s.start:.1f} | {s.text}" for s in segments]
    transcript = "\n".join(transcript_lines)
    # Gemini 2.0 Flash free tier handles ~1M tokens but be polite — cap input.
    if len(transcript) > 120_000:
        transcript = transcript[:120_000]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = HIGHLIGHT_PROMPT.format(
        n=n, min_len=min_len, max_len=max_len,
        video_title=video_title, transcript=transcript,
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    data = json.loads(resp.text)
    clips = data.get("clips", [])[:n]
    # sanity-clamp lengths
    out = []
    for c in clips:
        start = max(0.0, float(c["start"]))
        end = float(c["end"])
        if end - start < min_len:
            continue
        if end - start > max_len:
            end = start + max_len
        out.append({
            "start": start, "end": end,
            "title": c.get("title", "")[:90],
            "reason": c.get("reason", ""),
        })
    return out

# ---------- captions (ASS subtitle file, burned in) ----------

def words_in_range(segments: list[Segment], start: float, end: float) -> list[Word]:
    words: list[Word] = []
    for seg in segments:
        for w in seg.words:
            if w.end < start or w.start > end:
                continue
            words.append(Word(max(w.start, start), min(w.end, end), w.text.strip()))
    return words

def _ass_ts(t: float) -> str:
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60);   t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))
    if cs == 100:
        s += 1; cs = 0
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def build_caption_file(
    words: list[Word], clip_start: float, out_path: Path,
    chunk_size: int = 3,
) -> None:
    """Bouncy 2-3-word phrase captions, big bold center-bottom."""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,84,&H00FFFFFF,&H00000000,&H64000000,1,0,1,5,2,2,60,60,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for i in range(0, len(words), chunk_size):
        group = words[i:i + chunk_size]
        if not group:
            continue
        s = group[0].start - clip_start
        e = group[-1].end - clip_start
        text = " ".join(w.text for w in group).upper()
        # escape ASS-special chars
        text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
        lines.append(
            f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Default,,0,0,0,,{text}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ---------- ffmpeg cut + crop ----------

def cut_clip(
    src: Path, dst: Path, start: float, end: float, captions: Path,
) -> None:
    duration = end - start
    # Strategy: seek to start (-ss before -i for fast seek), then re-encode with
    # crop to 9:16 (centered vertical slice), scale to 1080x1920, burn ASS subs.
    # Captions filter takes the .ass file.
    vf = (
        f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',"
        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
        f"ass={captions.as_posix()}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(src),
        "-t", f"{duration:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)

# ---------- youtube upload ----------

def youtube_client():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(GoogleAuthRequest())
    return build("youtube", "v3", credentials=creds)

def upload_short(
    yt, file_path: Path, title: str, description: str, tags: list[str],
) -> str:
    title = (title or "Clip").strip()
    if "#shorts" not in title.lower() and len(title) < 95:
        title = f"{title} #Shorts"
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4500],
            "tags": tags[:15],
            "categoryId": "24",  # Entertainment; change in config if you like
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(file_path), mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _status, response = req.next_chunk()
    return response["id"]

# ---------- driver ----------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"

def process_video(
    video: dict, segments_cache_dir: Path, cfg: dict, yt,
) -> int:
    """Process one source video end-to-end. Returns number of clips uploaded."""
    print(f"[+] Processing {video['id']}: {video['title']}")
    video_dir = WORK / video["id"]
    video_dir.mkdir(parents=True, exist_ok=True)

    src = download_video(video["url"], video_dir)
    print(f"    downloaded -> {src.name}")

    segments = transcribe(src)
    print(f"    transcribed: {len(segments)} segments")

    clips = pick_highlights(
        segments, video["title"],
        n=cfg["clips_per_video"],
        min_len=cfg["min_clip_seconds"],
        max_len=cfg["max_clip_seconds"],
    )
    print(f"    Gemini picked {len(clips)} clips")

    uploaded = 0
    for i, clip in enumerate(clips, 1):
        out_mp4 = video_dir / f"clip_{i}_{slugify(clip['title'])}.mp4"
        ass = video_dir / f"clip_{i}.ass"
        words = words_in_range(segments, clip["start"], clip["end"])
        build_caption_file(words, clip["start"], ass)
        cut_clip(src, out_mp4, clip["start"], clip["end"], ass)
        print(f"    [{i}/{len(clips)}] cut {out_mp4.name} ({clip['end']-clip['start']:.1f}s)")

        desc = (
            f"{clip['title']}\n\n"
            f"From: {video['title']}\n"
            f"{video['url']}\n\n"
            f"#Shorts {' '.join('#' + t for t in cfg.get('hashtags', []))}"
        )
        try:
            vid_id = upload_short(yt, out_mp4, clip["title"], desc, cfg.get("hashtags", []))
            print(f"        uploaded -> https://youtu.be/{vid_id}")
            uploaded += 1
        except Exception as e:
            print(f"        upload FAILED: {e}", file=sys.stderr)
            # If quota exceeded, stop trying further uploads today.
            if "quotaExceeded" in str(e):
                print("    Quota exceeded — stopping uploads for today.")
                break
    return uploaded

def main() -> int:
    WORK.mkdir(exist_ok=True)
    cfg = load_config()
    state = load_state()
    processed = set(state["processed_video_ids"])

    yt = youtube_client()
    total_uploaded = 0
    daily_cap = cfg.get("max_uploads_per_day", 5)

    for ch in cfg["channels"]:
        if total_uploaded >= daily_cap:
            print(f"[stop] hit daily cap of {daily_cap} uploads")
            break
        try:
            latest = latest_video_for_channel(ch)
        except Exception as e:
            print(f"[warn] couldn't fetch latest for {ch}: {e}", file=sys.stderr)
            continue
        if not latest:
            continue
        if latest["id"] in processed:
            print(f"[-] {ch}: latest {latest['id']} already processed")
            continue
        if latest["duration"] and latest["duration"] < cfg["min_source_seconds"]:
            print(f"[-] {ch}: latest too short ({latest['duration']}s)")
            continue
        if latest["duration"] and latest["duration"] > cfg["max_source_seconds"]:
            print(f"[-] {ch}: latest too long ({latest['duration']}s)")
            continue

        # cap how many we'll try from this video so we don't blow the daily cap
        remaining = daily_cap - total_uploaded
        cfg_for_video = {**cfg, "clips_per_video": min(cfg["clips_per_video"], remaining)}
        try:
            n = process_video(latest, WORK, cfg_for_video, yt)
            total_uploaded += n
            processed.add(latest["id"])
            state["processed_video_ids"] = list(processed)[-500:]  # keep recent only
            save_state(state)
        except Exception as e:
            print(f"[error] processing {latest['id']}: {e}", file=sys.stderr)
            # don't mark as processed — we'll retry next run
        finally:
            # tidy up to keep runner disk small
            shutil.rmtree(WORK / latest["id"], ignore_errors=True)

    print(f"[done] uploaded {total_uploaded} clip(s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
