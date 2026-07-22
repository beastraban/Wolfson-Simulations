#!/usr/bin/env python3
"""Pass 2: capture the actual journal-ref strings for gen-ph and gr-qc
so venues can be inspected rather than merely counted. Checkpointed."""
import urllib.request, time, re, json, os, sys

CKPT = "/home/claude/venues_ckpt.json"
CATS = ["physics.gen-ph", "gr-qc"]
YEARS = [2020, 2021, 2022]
PER_MONTH = 200
DELAY = 3.0

def load():
    if os.path.exists(CKPT):
        return json.load(open(CKPT))
    return {"done": {}, "refs": []}

def save(s):
    tmp = CKPT + ".tmp"
    json.dump(s, open(tmp, "w"))
    os.replace(tmp, CKPT)

def fetch(cat, y, m):
    m2, y2 = (m + 1, y) if m < 12 else (1, y + 1)
    q = (f"cat:{cat}+AND+submittedDate:"
         f"[{y}{m:02d}010000+TO+{y2}{m2:02d}010000]")
    url = (f"http://export.arxiv.org/api/query?search_query={q}"
           f"&start=0&max_results={PER_MONTH}")
    for a in range(4):
        try:
            return urllib.request.urlopen(url, timeout=90).read().decode()
        except Exception as e:
            sys.stderr.write(f"  retry {a+1}: {e}\n")
            time.sleep(10 * (a + 1))
    return None

JREF = re.compile(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", re.S)

def main():
    s = load()
    jobs = [(c, y, m) for c in CATS for y in YEARS for m in range(1, 13)]
    todo = [k for k in jobs if f"{k[0]}|{k[1]}|{k[2]}" not in s["done"]]
    print(f"{len(todo)} cohorts remaining", flush=True)
    for cat, y, m in todo:
        xml = fetch(cat, y, m)
        if xml is None:
            continue
        for e in xml.split("<entry>")[1:]:
            mt = JREF.search(e)
            if mt:
                s["refs"].append({"cat": cat,
                                  "ref": " ".join(mt.group(1).split())})
        s["done"][f"{cat}|{y}|{m}"] = True
        save(s)
        print(f"{cat} {y}-{m:02d} -> {len(s['refs'])} refs", flush=True)
        time.sleep(DELAY)

if __name__ == "__main__":
    main()
