"""
Run this ONCE on your local machine to get a YouTube refresh token.
Saves you from doing OAuth dance inside GitHub Actions.

Usage:
    pip install google-auth-oauthlib
    python get_refresh_token.py path/to/client_secret.json
"""
import json
import sys
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def main(client_secret_path: str) -> None:
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    with open(client_secret_path) as f:
        cs = json.load(f)["installed"]
    print()
    print("=" * 70)
    print("Add these as repo secrets in GitHub (Settings -> Secrets -> Actions):")
    print("=" * 70)
    print(f"YT_CLIENT_ID      = {cs['client_id']}")
    print(f"YT_CLIENT_SECRET  = {cs['client_secret']}")
    print(f"YT_REFRESH_TOKEN  = {creds.refresh_token}")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python get_refresh_token.py path/to/client_secret.json")
        sys.exit(1)
    main(sys.argv[1])
