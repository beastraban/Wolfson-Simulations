#!/usr/bin/env python3
"""
Measure the fraction of arXiv papers that carry a journal-ref,
comparing physics.gen-ph against gr-qc over matched monthly cohorts.

journal-ref presence is a LOWER BOUND on publication: many authors
never update the field. The comparison between categories is the
meaningful quantity, not the absolute rate.

Checkpointed: safe to kill and restart.
"""
import urllib.request, urllib.error, time, re, json, os, sys

CKPT = "/home/claude/genph_ckpt.json"
CATS = ["physics.gen-ph", "gr-qc"]
YEARS = [2020, 2021, 2022]
PER_MONTH = 200          # cap per category-month
DELAY = 3.0              # arXiv API courtesy delay

def load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {"done": {}, "rows": []}

def save(state):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CKPT)

def fetch(cat, y, m):
    m2 = m + 1
    y2 = y
    if m2 == 13:
        m2, y2 = 1, y + 1
    lo = f"{y}{m:02d}010000"
    hi = f"{y2}{m2:02d}010000"
    q = f"cat:{cat}+AND+submittedDate:[{lo}+TO+{hi}]"
    url = (f"http://export.arxiv.org/api/query?search_query={q}"
           f"&start=0&max_results={PER_MONTH}")
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=90).read().decode()
        except Exception as e:
            sys.stderr.write(f"  retry {attempt+1} ({cat} {y}-{m:02d}): {e}\n")
            time.sleep(10 * (attempt + 1))
    return None

def parse(xml):
    """Return (n_entries, n_with_journal_ref)."""
    entries = xml.split("<entry>")[1:]
    n = len(entries)
    j = sum(1 for e in entries if "arxiv:journal_ref" in e)
    return n, j

def main():
    state = load()
    jobs = [(c, y, m) for c in CATS for y in YEARS for m in range(1, 13)]
    todo = [k for k in jobs if f"{k[0]}|{k[1]}|{k[2]}" not in state["done"]]
    print(f"{len(jobs)} cohorts total, {len(todo)} remaining", flush=True)

    for cat, y, m in todo:
        key = f"{cat}|{y}|{m}"
        xml = fetch(cat, y, m)
        if xml is None:
            print(f"FAILED {key}", flush=True)
            continue
        n, j = parse(xml)
        state["done"][key] = True
        state["rows"].append({"cat": cat, "year": y, "month": m,
                              "n": n, "with_ref": j})
        save(state)
        print(f"{cat:15s} {y}-{m:02d}  n={n:4d}  journal-ref={j:4d}"
              f"  {100*j/n if n else 0:5.1f}%", flush=True)
        time.sleep(DELAY)

    print("\n=== TOTALS ===", flush=True)
    for c in CATS:
        rows = [r for r in state["rows"] if r["cat"] == c]
        N = sum(r["n"] for r in rows)
        J = sum(r["with_ref"] for r in rows)
        print(f"{c:15s} sampled={N:6d}  with journal-ref={J:6d}"
              f"  = {100*J/N if N else 0:.1f}%", flush=True)

if __name__ == "__main__":
    main()
