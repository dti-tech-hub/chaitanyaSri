# Submission checklist

Work top to bottom. Items marked **[blocker]** must be done before submitting; the rest are
quality improvements.

## A. Metadata and authorship

- [ ] **[blocker]** Replace `[Author 1]`, `[Author 2]`, `[Author 3]`, `[Department]`,
      `[Institution]`, `[City]`, `[Country]` and the email placeholders on page 1.
- [ ] **[blocker]** Replace `[conference name]` and `[track]` with the actual venue.
- [ ] Confirm author order, corresponding author, ORCID iDs (if the venue requires them).
- [ ] Add funding and acknowledgements (none are currently claimed).
- [ ] Declare conflicts of interest if the venue requires it.
- [ ] Decide whether the paper is double-blind. If yes, see Section D.

## B. Numbers and claims

- [ ] **[blocker]** Re-run `python conference_package/scripts/make_paper_assets.py` and confirm
      the paper's tables match `tables/*.md`.
- [ ] Check these headline values against `TRACEABILITY.md`:
      - full test: 0.985 / 0.982 / 0.984 / 0.989 / 0.660
      - screened: 0.941 / 0.956 / 0.949 / 0.953 / 0.636
      - operating point: 222/4/4 (full), 129/4/4 (screened)
      - leakage: 77 committed, 81 dHash, 87 union, 56 kept, 133 boxes
      - control: 10.00→3.50 s, 26.00→3.50 s, 9.71→8.97 s, 28.17→52.61 s, 1,704→1,710 veh/h
      - per-event deltas: −4.5, +3.5, +3.5, −22.5, −12.5 s
- [ ] **[blocker]** Confirm the abstract does not claim a real-world safety, response-time,
      emissions or congestion benefit. It currently does not; do not add one in editing.
- [ ] Confirm the paper never calls the controller "deployed", "field-tested" or "validated on
      real traffic".
- [ ] Decide whether the screened split is primary (recommended). If not, move the ablation to
      a supplementary note and explain why.
- [ ] If you add a new experiment, add it to `TRACEABILITY.md` and regenerate assets.
- [ ] Do not add p-values to the deterministic five-EV simulation. If a reviewer asks, explain
      that the simulation is deterministic and the sample is too small for inference.

## C. Formatting (IEEE conference, two column)

- [ ] **[blocker]** Convert to the venue's template (IEEEtran `conference` is assumed).
- [ ] **[blocker]** Check the page limit. The draft is ~6,660 words including tables and
      references (~9 pages at 10 pt two column). If the limit is 6 pages, cut in this order:
      1. Appendix B (reproducibility) — move to a companion repository note;
      2. Section 2 (Related Work) — compress to one paragraph per theme;
      3. Merge Tables 3 and 4 into one table with "full / screened" column groups;
      4. Move Fig. 13 (confusion matrix) and Fig. 8/14 (PR curves) to supplementary;
      5. Shorten Section 7 (Discussion) by removing repeated caveats already in Section 8.
- [ ] Abstract ≤ 250 words (current draft is 234 words; verify after venue-specific edits).
- [ ] Figures: 300 dpi PNGs, column width 3.5 in / page width 7.16 in. Check axis text ≥ 7 pt
      after scaling. Figures 1–5 and 13 are generated; 6–12 and 14 are copied real plots.
- [ ] Tables: use the venue's booktabs style; keep the source note under each table.
- [ ] References: IEEE numeric style is already used. Verify [13] (FHWA-HOP-24-019, 2024) and
      the two dataset URLs resolve.
- [ ] Ensure every figure and table is cited in the text. Currently cited: Figs. 1–5, 9, 13;
      Tables 1–8, A1, A2. Figs. 6, 7, 8, 10, 11, 12, 14 are supplementary assets — either cite
      them or list them as supplementary in the caption set.
- [ ] Add line numbers if the venue requires them for review.

## D. Anonymisation (only if double-blind)

- [ ] Remove the repository URL and any GitHub handle.
- [ ] Remove Windows absolute paths (`B:\chaitanyasri\...`) from artifacts or the supplement.
- [ ] Remove Roboflow workspace names if they identify the authors (they currently appear in
      Section 5.1 and reference [12]).
- [ ] Remove acknowledgements/funding that identify the institution.
- [ ] Check figure metadata (PNG EXIF) for author names.

## E. Ethics, licences, attribution

- [ ] **[blocker]** Keep the simulated-prototype disclaimer in the abstract and Section 1.2.
- [ ] State that no human subjects were involved and no personal data were processed.
- [ ] Attribute both CC BY 4.0 datasets and note the modifications (merge, remap, dedup,
      rename). The required attribution text is in `docs/DATASETS.md` §1.1.
- [ ] Do not redistribute demo clips; they retain their original licences.
- [ ] Add an AI-use disclosure if the venue requires one.
- [ ] Add the data/code availability statement from `README.md` §8.

## F. Repository cleanup before attaching or linking the artifact

- [x] `results/report_tables.md` regenerated from the final 30-epoch evaluation (best epoch 22).
      Re-run `python scripts/generate_results.py` if any evaluation artifact changes.
- [ ] Fix `src/evaluation/detection_metrics.py` to copy `BoxPR_curve.png` →
      `precision_recall_curve.png` (currently looks for `PR_curve.png` and silently skips it).
- [ ] Remove or fill `paper/experiments/leakage_screened/` and `paper/experiments/sensitivity/`
      (currently empty).
- [ ] Reword `docs/METHODOLOGY.md` "disjoint by source image" to "disjoint by image, with a
      documented near-duplicate screen".
- [ ] Either produce a completed live run (`performance.csv` has only a header) or remove
      claims/links that imply live FPS/latency numbers exist.
- [x] Root README rewritten with the screened detector result and a clear data-availability
      section.
- [ ] Add `conference_package/` to the repository README (one line + link).

## G. Optional strengthening (reviewer-proofing, not required)

- [ ] Grouped train/test split by source sequence, then re-evaluate. This is the single
      highest-value addition: it would replace "near-duplicate-free" with "source-disjoint".
- [ ] Bootstrap 95% confidence intervals for mAP on the screened split (1,000 resamples).
- [ ] More seeds for the sweep (e.g., 10) and report the distribution of normal max wait, not
      only mean ± SD.
- [ ] Add a "no-priority" and a "green-path fast path" variant to quantify the 3.5 s cost.
- [ ] Add a queue-aware policy and report the normal-wait tail.
- [ ] ByteTrack comparison with MOTA/IDF1 on a small labelled sequence.

## H. Final pass

- [ ] Read the abstract aloud; every sentence must be supported by a table or figure.
- [ ] Search the text for "significant", "proves", "optimal", "safe" and soften unsupported uses.
- [ ] Check that "police", "fire engine" and "ambulance" counts in the abstract match Tables 3–4.
- [ ] Confirm units (s, veh/h, ms) and decimal places are consistent.
- [ ] Confirm figure numbers in captions match the files (`fig13` is Figure 13, etc.).
- [ ] Export the final PDF, check fonts are embedded, and inspect the figures at print size.
- [ ] Archive the exact PDF + this package + the git commit hash used for the submission.
