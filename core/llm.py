# core/llm.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# 1) LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Please add it to your .env file."
    )

client = OpenAI(api_key=api_key)


def complete(prompt):

    # ------------------------------------------------------------
    # 2) LIMIT INPUT LENGTH
    # (protect token usage)
    # ------------------------------------------------------------
    MAX_INPUT_CHARS = 500

    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS] + "... [truncated]"


    # ------------------------------------------------------------
    # 3) CALL LLM WITH OUTPUT LIMITED
    # ------------------------------------------------------------
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=120,      # Output limit
        temperature=0.7
    )

    return response.choices[0].message.content