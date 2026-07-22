#!/usr/bin/env python3
"""FULL CENSUS: every gr-qc and physics.gen-ph paper submitted 2020-2022.

Removes the 200/month sampling cap from the earlier pass, which was the
only serious methodological objection to the result.

Records, per paper: v1 submission year/month, journal-ref (if any),
journal publication year parsed from it.

Checkpointed at page granularity. Safe to kill and restart.
"""
import urllib.request, time, re, json, os, sys

CKPT = "/home/claude/census_ckpt.json"
CATS = ["physics.gen-ph", "gr-qc"]
YEARS = [2020, 2021, 2022]
PAGE = 500
DELAY = 3.0

JREF  = re.compile(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", re.S)
PUB   = re.compile(r"<published>(\d{4})-(\d{2})")
TOTAL = re.compile(r"opensearch:totalResults[^>]*>(\d+)<")
YR    = re.compile(r"(19[89]\d|20[0-3]\d)")
ID    = re.compile(r"<id>http://arxiv\.org/abs/([^<]+)</id>")

def load():
    if os.path.exists(CKPT):
        return json.load(open(CKPT))
    return {"pages": {}, "papers": {}}

def save(s):
    tmp = CKPT + ".tmp"
    json.dump(s, open(tmp, "w"))
    os.replace(tmp, CKPT)

def query(cat, y, m, start, n):
    m2, y2 = (m + 1, y) if m < 12 else (1, y + 1)
    q = (f"cat:{cat}+AND+submittedDate:"
         f"[{y}{m:02d}010000+TO+{y2}{m2:02d}010000]")
    url = (f"http://export.arxiv.org/api/query?search_query={q}"
           f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=ascending")
    for a in range(5):
        try:
            return urllib.request.urlopen(url, timeout=120).read().decode()
        except Exception as e:
            sys.stderr.write(f"   retry {a+1} {cat} {y}-{m:02d}@{start}: {e}\n")
            time.sleep(8 * (a + 1))
    return None

def harvest(xml, cat, store):
    added = 0
    for e in xml.split("<entry>")[1:]:
        im, pm = ID.search(e), PUB.search(e)
        if not (im and pm):
            continue
        aid = im.group(1).split("v")[0]
        if aid in store:
            continue
        jm = JREF.search(e)
        ref = " ".join(jm.group(1).split()) if jm else None
        jy = None
        if ref:
            ys = YR.findall(ref)
            if ys:
                jy = int(max(ys))
        store[aid] = {"cat": cat, "ay": int(pm.group(1)),
                      "am": int(pm.group(2)), "ref": ref[:90] if ref else None,
                      "jy": jy}
        added += 1
    return added

def main():
    s = load()
    for cat in CATS:
        for y in YEARS:
            for m in range(1, 13):
                start = 0
                total = None
                while True:
                    pk = f"{cat}|{y}|{m}|{start}"
                    if pk in s["pages"]:
                        start += PAGE
                        if total is not None and start >= total:
                            break
                        if s["pages"][pk].get("last"):
                            break
                        continue
                    xml = query(cat, y, m, start, PAGE)
                    if xml is None:
                        print(f"FAIL {pk}", flush=True)
                        break
                    tm = TOTAL.search(xml)
                    total = int(tm.group(1)) if tm else 0
                    n = harvest(xml, cat, s["papers"])
                    got = len(xml.split("<entry>")) - 1
                    last = (start + PAGE >= total) or got == 0
                    s["pages"][pk] = {"got": got, "total": total, "last": last}
                    save(s)
                    print(f"{cat:15s} {y}-{m:02d} start={start:5d} "
                          f"got={got:4d}/{total:5d} new={n:4d} "
                          f"cum={len(s['papers'])}", flush=True)
                    time.sleep(DELAY)
                    if last:
                        break
                    start += PAGE
    print(f"\nCENSUS COMPLETE: {len(s['papers'])} papers", flush=True)

if __name__ == "__main__":
    main()
