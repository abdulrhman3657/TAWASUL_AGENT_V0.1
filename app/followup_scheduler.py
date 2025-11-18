# app/followup_scheduler.py

"""
Scheduler that periodically runs the follow-up worker.

Usage:
    python -m app.followup_scheduler

It will:
    - Every N minutes/hours, call run_followup_once()
    - That function finds stale open tickets and queues follow-up emails.
"""

import time
import schedule  # pip install schedule

from .followup_worker import run_followup_once


def job():
    print("Running follow-up job...")
    run_followup_once()
    print("Follow-up job finished.\n")


def main():
    # 🔹 Choose how often you want to run it:
    # Every 30 minutes:
    # schedule.every(30).minutes.do(job)

    # Every hour:
    schedule.every().day.do(job)

    # For testing, you can do:
    # schedule.every(1).minutes.do(job)

    print("Follow-up scheduler started. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
