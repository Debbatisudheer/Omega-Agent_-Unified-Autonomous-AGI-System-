# core/llm.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if API_KEY:
    client = OpenAI(api_key=API_KEY)


def complete(prompt):

    MAX_INPUT_CHARS = 500

    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS] + "... [truncated]"

    # Offline fallback
    if client is None:
        return "⚠️ AI research summary unavailable (offline mode)."

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI service unavailable right now. Please try again later."