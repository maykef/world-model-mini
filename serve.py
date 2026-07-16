#!/usr/bin/env python3
"""Static file server for the sim + /stats endpoint (GPU/CPU/RAM gauges).
Run by the hillsim systemd user service; proxied by tailscale serve on :8443."""
import json, os, subprocess, time, http.server

DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {"t": 0.0, "data": {}}
_cpu_prev = None


def cpu_pct():
    global _cpu_prev
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle, total = vals[3] + vals[4], sum(vals)
    if _cpu_prev is None:
        _cpu_prev = (idle, total)
        return 0.0
    di, dt = idle - _cpu_prev[0], total - _cpu_prev[1]
    _cpu_prev = (idle, total)
    return round(100 * (1 - di / dt), 1) if dt > 0 else 0.0


def ram():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return round((m["MemTotal"] - m["MemAvailable"]) / 1048576, 1), round(m["MemTotal"] / 1048576, 1)


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], timeout=3).decode().strip().split(", ")
        return {"util_pct": float(out[0]), "vram_used_gb": round(float(out[1]) / 1024, 1),
                "vram_total_gb": round(float(out[2]) / 1024, 1), "temp_c": float(out[3]),
                "power_w": float(out[4])}
    except Exception:
        return None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIR, **k)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/stats":
            now = time.time()
            if now - _cache["t"] > 1.5:
                used, total = ram()
                _cache["data"] = {"gpu": gpu(), "cpu_pct": cpu_pct(),
                                  "ram_used_gb": used, "ram_total_gb": total}
                _cache["t"] = now
            body = json.dumps(_cache["data"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


http.server.ThreadingHTTPServer(("127.0.0.1", 8388), Handler).serve_forever()
