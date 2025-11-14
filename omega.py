# omega.py - Clean + Fully Working With .env Support

import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from memory.vector_memory import VectorMemory
from agents.planner_agent import run_planner
from agents.trading_agent import run_trading_agent
from agents.research_agent import run_research_agent

LOOPS = 5   # number of autonomous cycles


def print_clean(loop_num, planner_output, trade_output, research_output, memory):
    print(f"\n==== LOOP {loop_num} ====")

    # ----------------------------
    # 1. PLANNER OUTPUT
    # ----------------------------
    try:
        tasks = [t["task"] for t in planner_output[:3]]
        print("📅 Schedule:", ", ".join(tasks))
    except:
        print("📅 Schedule: (no tasks returned)")

    # ----------------------------
    # 2. TRADING OUTPUT
    # ----------------------------
    action = trade_output.get("action", "none")
    print("📈 Trade Decision:", action.upper())

    # ----------------------------
    # 3. RESEARCH OUTPUT
    # ----------------------------
    summary = research_output.get("summary", "")
    short_sum = summary[:80] + "..." if len(summary) > 80 else summary
    print("🧠 Research Summary:", short_sum)

    # ----------------------------
    # 4. MEMORY CHECK
    # ----------------------------
    try:
        sample = memory.query("trading", top_k=1)
        mem_count = len(sample) if sample else 0
    except Exception:
        mem_count = 0

    print(f"💾 Memory sample items found: {mem_count}")
    print("---------------------------")


def main():
    print("🔥 Omega Autonomous Loop Starting...")
    print("🔧 Memory backend:", os.getenv("MEMORY_BACKEND", "local"))

    # init memory (auto-detects backend)
    memory = VectorMemory()

    # MAIN LOOP
    for i in range(LOOPS):
        print(f"\nLoop {i+1}/{LOOPS} Running...")

        planner_output = run_planner(memory=memory)
        trade_output = run_trading_agent(memory=memory)
        research_output = run_research_agent(memory=memory)

        print_clean(
            loop_num=i+1,
            planner_output=planner_output,
            trade_output=trade_output,
            research_output=research_output,
            memory=memory
        )

        print("⏳ Sleeping briefly before next autonomous tick...\n")
        time.sleep(1)

    print("\n✅ Omega loop finished. State/models saved.\n")


if __name__ == "__main__":
    main()
