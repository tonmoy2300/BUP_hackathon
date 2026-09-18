import json
import time
import urllib.request


def post(fname):
    with open(fname, "rb") as f:
        data = f.read()
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/optimize-energy",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=60)
        body = json.loads(r.read())
        elapsed = time.time() - t0
        print(f"=== {fname} ===")
        print(f"STATUS: 200  TIME: {elapsed:.2f}s")
        print("directive_interpretation:")
        for d in body["directive_interpretation"]:
            ni = d["note_index"]
            dt = d["directive_type"]
            ap = d["applies"]
            adj = d["structured_adjustment"]
            exp = d["explanation"]
            print(f"  [{ni}] {dt} applies={ap} adj={adj}")
            print(f"       {exp}")
        print(f"total_grid_kwh: {body['total_grid_kwh']}")
        print(f"total_cost_bdt: {body['total_cost_bdt']}")
        print(f"peak_grid_kwh:  {body['peak_grid_kwh']}")
        print()
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"=== {fname} ===")
        print(f"ERROR ({elapsed:.2f}s): {exc}")
        try:
            print(exc.read())
        except Exception:
            pass
        print()


for fn in ["sample01.json", "sample02_real.json", "paraphrase01.json"]:
    post(fn)
