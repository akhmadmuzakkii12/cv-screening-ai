import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JOOBLE_API_KEY")

def get_jobs():
    if not API_KEY:
        print("❌ API Key tidak ditemukan. Pastikan ada di file .env")
        return []
    
    url = f"https://jooble.org/api/{API_KEY}"
    payload = {
        "keywords": "data scientist",
        "location": "Indonesia",
        "page": 1
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("jobs", [])
        else:
            print(f"⚠️ Gagal ambil data Jooble API. Status: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error saat ambil data API: {e}")
        return []

# Testing
if __name__ == "__main__":
    jobs = get_jobs()
    print(jobs[:3])  # print 3 job pertama
