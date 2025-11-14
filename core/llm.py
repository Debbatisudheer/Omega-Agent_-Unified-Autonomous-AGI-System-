# core/llm.py

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def complete(prompt):
    # ------------------------------------------------------------
    # 1) LIMIT INPUT LENGTH
    #    (avoid long prompts, protect tokens)
    # ------------------------------------------------------------
    MAX_INPUT_CHARS = 500
    if len(prompt) > MAX_INPUT_CHARS:
        prompt = prompt[:MAX_INPUT_CHARS] + "... [truncated]"

    # ------------------------------------------------------------
    # 2) CALL LLM WITH OUTPUT LIMITED TO 120 TOKENS
    # ------------------------------------------------------------
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,   # <-- Output capped at 120 tokens
        temperature=0.7
    )

    return response.choices[0].message.content
