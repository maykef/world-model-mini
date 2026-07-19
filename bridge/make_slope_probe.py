#!/usr/bin/env python3
"""make_slope_probe.py — generate a slope-probe dataset for `mind_driver.py --slope-probe`.

Only the WebGL sim can produce the shaded terrain frames, so this drives headless Chromium
against `index.html?selftest=slopeset&render=<config>`, which renders a set of deterministic
(seed-4242) egocentric viewpoints on perlin terrain and reports, for each, ground-truth grade %
ahead/left/right at 5 m and 15 m. The SAME viewpoints are used for every render config (only the
shader differs), so per-config VLM accuracy is directly comparable.

Output layout (consumed by --slope-probe):
  <out>/manifest.jsonl                 # one JSON per frame: {image, render_config, pos, heading_deg, grade_pct}
  <out>/<config>/frame_NNN.jpg

Snap Chromium is confined to $HOME, so index.html is staged into a temp dir there.
Usage: venv/bin/python bridge/make_slope_probe.py --out data/slopeset --samples 12
"""
import argparse, base64, json, os, re, shutil, subprocess, tempfile

CHROMIUM = "/snap/bin/chromium"
CONFIGS = ["naturalistic", "bands", "bands+contours"]
HERE = os.path.dirname(os.path.abspath(__file__))


def capture(base_url, config, samples, timeout):
    url = f"{base_url}?selftest=slopeset&render={config.replace('+', '%2B')}&samples={samples}"
    dom = subprocess.run(
        [CHROMIUM, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         "--virtual-time-budget=15000", "--dump-dom", url],
        capture_output=True, text=True, timeout=timeout).stdout
    m = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    if not m:
        raise RuntimeError(f"no slopeset output for render={config} (is index.html current?)")
    txt = (m.group(1).replace("&amp;", "&").replace("&lt;", "<")
           .replace("&gt;", ">").replace("&quot;", '"'))
    return json.loads(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset output directory")
    ap.add_argument("--samples", type=int, default=12, help="viewpoints per render config")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma list of render configs")
    ap.add_argument("--index", default=os.path.join(os.path.dirname(HERE), "index.html"))
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="slopeset_", dir=os.path.expanduser("~"))  # snap-confined to $HOME
    shutil.copy(args.index, os.path.join(stage, "index.html"))
    base_url = f"file://{stage}/index.html"
    os.makedirs(args.out, exist_ok=True)
    total = 0
    try:
        with open(os.path.join(args.out, "manifest.jsonl"), "w") as mf:
            for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
                data = capture(base_url, config, args.samples, args.timeout)
                cdir_name = config.replace("+", "_")
                os.makedirs(os.path.join(args.out, cdir_name), exist_ok=True)
                for i, s in enumerate(data["samples"]):
                    rel = os.path.join(cdir_name, f"frame_{i:03d}.jpg")
                    with open(os.path.join(args.out, rel), "wb") as f:
                        f.write(base64.b64decode(s["jpg_b64"]))
                    mf.write(json.dumps({"image": rel, "render_config": config,
                                         "pos": s["pos"], "heading_deg": s["heading_deg"],
                                         "grade_pct": s["gt"]}) + "\n")
                    total += 1
                print(f"  {config}: {len(data['samples'])} frames", flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"wrote {total} samples -> {args.out}/manifest.jsonl", flush=True)


if __name__ == "__main__":
    main()
