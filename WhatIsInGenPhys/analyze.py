#!/usr/bin/env python3
"""
Reproduces every figure quoted in "What Is Actually In physics.gen-ph".

Usage:  python3 analyze.py [census_2020_2022.json]

Reads the census produced by census.py and prints all published numbers.
The FLAGSHIP and MDPI lists below are judgment calls, stated explicitly
so they can be argued with. Edit them and re-run to test sensitivity.
"""
import json, sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "census_2020_2022.json"

FLAGSHIP = [
    "phys. rev. d", "physical review d", "phys.rev.d", "phys rev d",
    "phys. rev. lett", "physical review letters", "phys.rev.lett",
    "class. quantum grav", "classical and quantum gravity", "class.quant.grav",
    "jhep", "j. high energ", "journal of high energy",
    "jcap", "j. cosmol. astropart",
    "mnras", "mon. not", "monthly notices",
    "astrophys. j", "astrophys.j", "astrophysical journal",
    "physics letters b", "phys. lett. b", "phys.lett.b",
    "nucl. phys. b", "nuclear physics b",
    "nature",
]

# Substrings that would otherwise be caught by the list above but which are
# NOT flagship venues. Checked by hand against the corpus.
EXCLUDE = [
    "astrophysics and space science",
    "bulgarian astronomical journal",
]

MDPI = ["symmetry", "universe", "entropy", "galaxies",
        "particles", "axioms", "mathematics"]

CATS = ["physics.gen-ph", "gr-qc"]


def is_flagship(ref):
    r = ref.lower()
    if any(x in r for x in EXCLUDE):
        return False
    return any(k in r for k in FLAGSHIP)


def is_mdpi(ref):
    r = ref.lower().strip()
    return any(r.startswith(k) for k in MDPI)


def main():
    papers = json.load(open(PATH))
    if "papers" in papers:
        papers = papers["papers"]
    P = list(papers.values())

    print("=" * 74)
    print("CENSUS: arXiv submissions 2020-2022")
    print("=" * 74)

    summary = {}
    for cat in CATS:
        R = [p for p in P if p["cat"] == cat]
        pub = [p for p in R if p["ref"]]
        dated = [p for p in pub if p["jy"]]
        after = [p for p in dated if 1 <= p["ay"] - p["jy"] <= 3]
        retro = [p for p in dated if p["ay"] - p["jy"] > 3]
        flag = [p for p in pub if is_flagship(p["ref"])]
        mdpi = [p for p in pub if is_mdpi(p["ref"])]

        summary[cat] = dict(n=len(R), pub=len(pub), dated=len(dated),
                            after=len(after), retro=len(retro),
                            flag=len(flag), mdpi=len(mdpi))

        print(f"\n{cat}")
        print(f"  papers submitted                 {len(R):7d}")
        print(f"  carrying a journal-ref           {len(pub):7d}   "
              f"{100*len(pub)/len(R):5.1f}%")
        print(f"  ...in flagship venues            {len(flag):7d}   "
              f"{100*len(flag)/len(pub):5.1f}%  (of published)")
        print(f"  ...in MDPI-family journals       {len(mdpi):7d}   "
              f"{100*len(mdpi)/len(pub):5.1f}%  (of published)")
        print(f"  deposited 1-3 yr AFTER journal   {len(after):7d}   "
              f"{100*len(after)/len(dated):5.1f}%  (of published)")
        print(f"  deposited >3 yr after (retro)    {len(retro):7d}   "
              f"{100*len(retro)/len(dated):5.1f}%  (of published)")

    g, q = summary["physics.gen-ph"], summary["gr-qc"]
    ra = (100*g["after"]/g["dated"]) / (100*q["after"]/q["dated"])
    rf = (100*q["flag"]/q["pub"]) / (100*g["flag"]/g["pub"])
    print("\n" + "=" * 74)
    print(f"post-publication deposit, gen-ph : gr-qc   =  {ra:.1f}x")
    print(f"flagship rate,            gr-qc : gen-ph   =  {rf:.1f}x")

    print("\n" + "=" * 74)
    print("THE READMITTED POPULATION (gen-ph, deposited 1-3 yr after journal)")
    print("=" * 74)
    R = [p for p in P if p["cat"] == "physics.gen-ph" and p["ref"] and p["jy"]]
    after = [p for p in R if 1 <= p["ay"] - p["jy"] <= 3]
    norm = [p for p in R if p["ay"] - p["jy"] <= 0]
    fa = sum(1 for p in after if is_flagship(p["ref"]))
    fn = sum(1 for p in norm if is_flagship(p["ref"]))
    print(f"  readmitted papers                {len(after):5d}   "
          f"flagship: {fa}  ({100*fa/len(after):.1f}%)")
    print(f"  normal-order papers              {len(norm):5d}   "
          f"flagship: {fn}  ({100*fn/len(norm):.1f}%)")
    print("\n  venues of the readmitted population:")
    for p in sorted(after, key=lambda x: x["ay"] - x["jy"]):
        print(f"    +{p['ay']-p['jy']}y  {p['ref'][:66]}")

    print("\n" + "=" * 74)
    print("TOP VENUES BY CATEGORY")
    print("=" * 74)
    import re
    for cat in CATS:
        refs = [p["ref"] for p in P if p["cat"] == cat and p["ref"]]
        names = [re.split(r"[,;]|\s\d{2,}|\svol|\sVol|\(\d{4}\)", r)[0].strip(" .")[:48]
                 for r in refs]
        print(f"\n  {cat}")
        for name, k in Counter(names).most_common(12):
            print(f"    {k:5d}  {name}")


if __name__ == "__main__":
    main()
