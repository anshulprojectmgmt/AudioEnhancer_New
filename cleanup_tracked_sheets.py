import csv
import os
import json
from datetime import datetime, timedelta
from google.oauth2 import service_account # Updated for cleaner env handling
from googleapiclient.discovery import build
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv() # Load variables from .env
LOG_FILE = '/home/ubuntu/AudioEnhancer_New/sheets_log.csv' # Use Absolute Path
DAYS_TO_KEEP = 1
# ---------------------

def delete_tracked_sheets():
    if not os.path.exists(LOG_FILE):
        print("No log file found. Nothing to clean.")
        return

    # 1. Load Credentials from ENV (Not file)
    json_creds = os.getenv("GOOGLE_SA_JSON") # <--- CHECK THIS NAME IN YOUR .ENV
    
    if not json_creds:
        print("❌ Error: GOOGLE_SA_JSON not found in .env")
        return

    try:
        # Parse the string into a dictionary
        creds_dict = json.loads(json_creds)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, 
            scopes=['https://www.googleapis.com/auth/drive']
        )
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
