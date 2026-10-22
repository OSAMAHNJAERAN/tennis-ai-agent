"""Read-only check of this task's frame-review process, without other commands."""
import json
import psutil
for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_times", "create_time"]):
    try:
        command = process.info.get("cmdline") or []
        if any("scripts.data.review_racketvision_ball_training" in part for part in command):
            print(json.dumps({"pid": process.pid, "created": process.info["create_time"],
                              "rss_bytes": process.info["memory_info"].rss,
                              "cpu_seconds": sum(process.info["cpu_times"][:2])}))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
