import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("JOOBLE_API_KEY")
print("API Key terbaca:", API_KEY)

url = f"https://jooble.org/api/{API_KEY}"
payload = {"keywords": "data scientist", "location": "Indonesia"}

res = requests.post(url, json=payload)
print("Status:", res.status_code)
print("Response:", res.text[:500])  # tampilkan 500 karakter pertama
