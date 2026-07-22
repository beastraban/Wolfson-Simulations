# What is actually in physics.gen-ph

Full census of arXiv submissions to `physics.gen-ph` and `gr-qc`, 2020–2022,
with journal publication metadata. Accompanies the post
*"What Is Actually In physics.gen-ph"*.

Data snapshot taken 22 July 2026. arXiv `journal-ref` fields are added by
authors over time, so re-running `census.py` later will give slightly
higher publication rates than those quoted below. That is why the snapshot
is included.

## Results

| | gen-ph | gr-qc |
|---|---|---|
| papers submitted 2020–2022 | 869 | 17,897 |
| carrying a `journal-ref` | 44.6% | 51.8% |
| of those, in flagship venues | 4.9% | 56.2% |
| of those, in MDPI-family journals | 11.6% | 5.2% |
| deposited 1–3 yr **after** journal publication | 10.1% | 1.9% |
| deposited >3 yr after (retro-archiving) | 2.1% | 0.5% |

Post-publication deposit rate is **5.2× higher** in gen-ph.
Flagship rate is **11.5× higher** in gr-qc.

Of the 39 gen-ph papers deposited after publication, **zero** appeared in a
flagship venue.

## Reproducing

```
python3 census.py        # ~30 min, re-fetches from the arXiv API, checkpointed
python3 analyze.py       # prints every figure quoted in the post
```

`analyze.py census_2020_2022.json` runs against the included snapshot instead
of a fresh fetch.

## Files

| file | what it is |
|---|---|
| `census.py` | full census with pagination — authoritative collector |
| `analyze.py` | regenerates every published figure; venue lists are editable |
| `census_2020_2022.json` | snapshot, 18,766 papers (1.8 MB) |
| `genph_study.py` | earlier pass: journal-ref counts on a 200/month sample |
| `genph_venues.py` | earlier pass: extracts journal-ref strings |
| `genph_order.py` | earlier pass: deposit-date vs publication-year |
| `raw_results.json` | outputs of the three sampled passes |

The three `genph_*` scripts are superseded by `census.py`. They are included
because the post claims the capped sample and the full census agree to within
one percentage point, and these are what that claim rests on.

## Caveats

`journal-ref` is **author-reported** and is a lower bound on publication;
reporting habits may differ between categories. This is the limitation that
cannot be fixed from outside.

"Flagship" is a keyword list (PRD, PRL, CQG, JHEP, JCAP, PLB, MNRAS, ApJ,
Nucl. Phys. B, Nature), not a ranking-body classification. It is defined at the
top of `analyze.py` together with an explicit exclusion list for near-misses
(*Astrophysics and Space Science*, *Bulgarian Astronomical Journal*). Edit and
re-run to test sensitivity.

Journal year is parsed from free-text `journal-ref`, so **same-year orderings
cannot be resolved** — the 10.1% figure is a floor. Resolving journal months
via Crossref would turn it into an estimate.

Window is three years. Categories are primary-category assignments.

## License

Data and code released CC0 / public domain. Corrections welcome — open an issue.
