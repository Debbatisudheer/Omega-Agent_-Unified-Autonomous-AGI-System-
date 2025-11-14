# agents/planner_agent.py

import json
import os
from datetime import datetime, time, timedelta
from dateutil import parser
from memory.vector_memory import VectorMemory

TASKS_FILE = "plm_tasks.json"

DEFAULT_TASKS = [
    {"id":"p1","title":"Deep Work - Project", "est_minutes":90, "priority":8, "deadline":None, "tags":["work"], "energy":"high"},
    {"id":"p2","title":"Market review", "est_minutes":30, "priority":6, "deadline":None, "tags":["finance"], "energy":"medium"},
    {"id":"p3","title":"Read research paper", "est_minutes":45, "priority":5, "deadline":None, "tags":["learning"], "energy":"low"},
]

class PlannerAgent:
    def __init__(self, memory: VectorMemory = None):
        self.memory = memory
        if not os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, "w") as f:
                json.dump(DEFAULT_TASKS, f, indent=2)
        with open(TASKS_FILE, "r") as f:
            self.tasks = json.load(f)
        self.alpha = 0.25

    def _save_tasks(self):
        with open(TASKS_FILE, "w") as f:
            json.dump(self.tasks, f, indent=2)

    def plan_day(self):
        # simple priority sort + schedule into working window
        now = datetime.now()
        work_start = datetime.combine(now.date(), time(9,0))
        start = max(now, work_start)
        schedule = []
        tasks_sorted = sorted(self.tasks, key=lambda t: t.get("priority",5), reverse=True)
        for t in tasks_sorted:
            est = int(t.get("est_minutes",30))
            end = start + timedelta(minutes=est)
            schedule.append({
                "task_id": t["id"],
                "title": t["title"],
                "start": start.isoformat(),
                "end": end.isoformat(),
                "est_minutes": est
            })
            start = end + timedelta(minutes=5)

        # write small summary into memory
        if self.memory:
            self.memory.add(
                "Planned day with tasks: " + ", ".join([s["title"] for s in schedule]),
                metadata={"type":"planner_summary"}
            )

        return schedule

    def update_after_task(self, task_id, actual_minutes):
        for t in self.tasks:
            if t["id"] == task_id:
                old = t.get("est_minutes", 30)
                new = int(round((1-self.alpha)*old + self.alpha*actual_minutes))
                t["est_minutes"] = new
                if self.memory:
                    self.memory.add(
                        f"Updated est for {t['title']} from {old} -> {new}",
                        metadata={"type":"planner_update"}
                    )
                self._save_tasks()
                return t
        return None


# ============================================================
# WRAPPER FUNCTION — REQUIRED BY omega.py
# ============================================================
def run_planner(memory=None):
    """
    Wrapper for Omega system.
    Accepts optional VectorMemory and returns cleaned schedule.
    """
    agent = PlannerAgent(memory=memory)
    schedule = agent.plan_day()

    clean_output = []
    for item in schedule:
        clean_output.append({
            "task": item["title"],
            "est_minutes": item["est_minutes"]
        })
    return clean_output
