import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1beta'}
)

print("AI se sawal pooch rahe hain...")

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Hello! My name is Syed Siraj Haider. Give me 2 lines of motivation for my Tech Mahindra interview."
)

print("\nAI ka jawab:")
print(response.text)