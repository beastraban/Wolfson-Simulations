#!/usr/bin/env python3
"""Pass 3: for papers carrying a journal-ref, compare the arXiv v1
submission date against the journal publication year.

Normal scholarly order: preprint first, journal later (arxiv_year <= jrnl_year).
The anomaly of interest: arxiv_year > jrnl_year, i.e. the paper was
deposited on arXiv only AFTER it had already been published.

Checkpointed."""
import urllib.request, time, re, json, os, sys

CKPT = "/home/claude/order_ckpt.json"
CATS = ["physics.gen-ph", "gr-qc"]
YEARS = [2020, 2021, 2022]
PER_MONTH = 200
DELAY = 3.0

def load():
    return json.load(open(CKPT)) if os.path.exists(CKPT) else {"done": {}, "recs": []}

def save(s):
    tmp = CKPT + ".tmp"; json.dump(s, open(tmp, "w")); os.replace(tmp, CKPT)

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
            sys.stderr.write(f"  retry {a+1}: {e}\n"); time.sleep(10 * (a + 1))
    return None

JREF = re.compile(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", re.S)
PUB  = re.compile(r"<published>(\d{4})-(\d{2})")
YR   = re.compile(r"(19[89]\d|20[0-3]\d)")

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
            jm, pm = JREF.search(e), PUB.search(e)
            if not (jm and pm):
                continue
            ref = " ".join(jm.group(1).split())
            yrs = YR.findall(ref)
            if not yrs:
                continue
            s["recs"].append({"cat": cat,
                              "arxiv_y": int(pm.group(1)),
                              "arxiv_m": int(pm.group(2)),
                              "jrnl_y": int(max(yrs)),
                              "ref": ref[:90]})
        s["done"][f"{cat}|{y}|{m}"] = True
        save(s)
        print(f"{cat} {y}-{m:02d} -> {len(s['recs'])} usable", flush=True)
        time.sleep(DELAY)

if __name__ == "__main__":
    main()
