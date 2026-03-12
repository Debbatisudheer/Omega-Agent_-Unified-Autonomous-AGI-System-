# core/llm.py

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

# Only create client if API key exists
client = OpenAI(api_key=API_KEY) if API_KEY else None


def complete(prompt):

    # ------------------------------------------------------------
    # 1) LIMIT INPUT LENGTH
    # ------------------------------------------------------------
    MAX_INPUT_CHARS = 500
    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS] + "... [truncated]"

    # ------------------------------------------------------------
    # 2) OFFLINE MODE (NO API KEY)
    # ------------------------------------------------------------
    if not API_KEY:
        return "⚠️ AI research module is currently offline."

    # ------------------------------------------------------------
    # 3) CALL OPENAI SAFELY
    # ------------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI service temporarily unavailable. Please try again later."