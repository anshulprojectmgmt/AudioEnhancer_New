import csv
import os
import json
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials  # ✅ Changed to standard Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv() # Load variables from .env
LOG_FILE = '/home/ubuntu/AudioEnhancer_New/sheets_log.csv'
TOKEN_FILE = '/home/ubuntu/AudioEnhancer_New/token.json' # ✅ Path to your fixed token
DAYS_TO_KEEP = 1
SCOPES = ['https://www.googleapis.com/auth/drive'] # We only need Drive scope for deletion
# ---------------------

def delete_tracked_sheets():
    if not os.path.exists(LOG_FILE):
        print("No log file found. Nothing to clean.")
        return

    # 1. Load Credentials from token.json (✅ NEW OAUTH METHOD)
    if not os.path.exists(TOKEN_FILE):
        print(f"❌ Error: Token file not found at {TOKEN_FILE}")
        return

    try:
        # This automatically handles the refresh token logic if needed
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        service = build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return

    # 2. Read the log
    rows_to_keep = []
    deleted_count = 0
    
    with open(LOG_FILE, "r") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    print(f"🔍 Checking {len(all_rows)} tracked sheets...")
    
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_KEEP)

    for row in all_rows:
        if len(row) < 2: continue

        sheet_id, created_at_str = row
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except ValueError:
            continue

        if created_at < cutoff_date:
            print(f"   🗑️ Deleting old sheet: {sheet_id}")
            try:
                service.files().delete(fileId=sheet_id).execute()
                print("      ✅ Deleted.")
                deleted_count += 1
            except Exception as e:
                if "404" in str(e):
                    print("      ⚠️ Already gone (404). Removing from log.")
                    deleted_count += 1
                else:
                    print(f"      ❌ Failed: {e}")
                    # Keep in list to try again later
                    rows_to_keep.append(row)
        else:
            # Keep new sheets
            rows_to_keep.append(row)

    # 3. REWRITE the log (Prevents infinite growth)
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_keep)

    print(f"\n✅ Cleanup Complete. {deleted_count} sheets removed from tracker.")

if __name__ == '__main__':
    delete_tracked_sheets()
