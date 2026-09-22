# Project report - chapter plan

A structure that matches what this codebase can actually support with evidence.
Each section names the artefact it should be written from, so nothing has to be
invented at writing time.

## 1. Introduction (4-5 pages)
Problem, motivation, scope, contributions, report organisation. State the scope
honestly: a local research prototype, not a mail gateway.

## 2. Literature review (8-12 pages)
Group by approach rather than chronology: content-based filtering (Naive Bayes
lineage), URL and lexical phishing detection, header/authentication-based
approaches, embedding and transformer methods, ensemble methods, explainability
in security. End with a comparison table (approach, data, reported metric,
limitation) and a paragraph naming the gap this project addresses: most published
work evaluates a single signal, in-corpus, without reporting false-positive cost.

## 3. System analysis (5-7 pages)
Existing systems, their limitations, functional and non-functional requirements,
feasibility. Cite `.env.example` for the configurable requirements.

## 4. System design (10-14 pages)
Architecture diagram from `docs/architecture.md`. DFD level 0/1, use-case, class
and sequence diagrams. Database schema from `database.py` (7 tables - reproduce
the DDL). Describe the abstention mechanism explicitly; it is a design decision
examiners will ask about.

## 5. Dataset (4-6 pages)
Each corpus: source, size, licence, citation, label mapping, what it does and
does not contain. De-duplication before splitting and why. Split strategy
(time-based when dates exist) and the held-out cross-corpus source. Class
distribution table from `training_report.json -> class_counts`.

## 6. Methodology - detection layers (12-16 pages)
One subsection per layer: what it computes, the feature list, why those features,
and its failure modes. Feature counts are in the code: 27 URL features
(`detectors/url.py:FEATURE_NAMES`), 44 tabular features (`ml/features.py`),
11 social-engineering categories.

## 7. Ensemble and risk scoring (6-8 pages)
Weighted fusion, logistic stacking, weight renormalisation over available layers,
fitted weights from `training_report.json -> fusion.weights`, band thresholds and
how E5 re-derives them under a stated cost model.

## 8. Implementation (8-10 pages)
Stack, module map, key code walkthroughs (parser hardening, abstention, promotion
gate), configuration, deployment steps from the README.

## 9. Experiments and results (12-18 pages) - the chapter that carries the project
Setup, then straight from `data/reports/experiments.md`:
E1 single layers, E1b lexical baselines, E2 fusion comparison, E3 ablation,
E4 cross-corpus, E5 operating points and cost sweep, E6 latency.
Include the confusion matrices and the figures written to `data/reports/`.
Discuss E4 candidly: cross-corpus performance is lower than in-corpus, and
explaining why is worth more marks than hiding it.

## 10. Explainability (4-6 pages)
Per-layer contributions, SHAP on the fusion model, lexical term attributions, a
worked example with screenshots of the email detail page. State the limitation:
attention weights are not used as explanations because attention is not causal
attribution.

## 11. Security and privacy analysis (4-6 pages)
Threat model for the tool itself: hostile input to the parser, OAuth token
storage, feedback poisoning and why retraining is gated, log pseudonymisation,
no URL fetching, no attachment execution. Adversarial pressure: what an attacker
changes to evade each layer, and which layers still fire.

## 12. Testing (3-5 pages)
Unit, integration, and the abstention regression test. Table of test names to
requirements.

## 13. Results discussion and limitations (4-5 pages)
Where it fails: 2003-era spam is easy, small brand list, no sandboxing, one
language, class imbalance, corpus age versus current phishing.

## 14. Conclusion and future work (3-4 pages)
Sandboxed attachment analysis, multilingual support, IMAP provider, federated or
on-device continual learning, per-user threshold calibration.

## 15. References (IEEE style)
Datasets and libraries included, not just papers.

## 16. Appendices
Selected source listings, full experiment JSON, installation transcript, viva
questions from `docs/viva.md`.

## Before submission
- Every number traced to a file in `data/reports/` or `models/v*/`.
- No table populated from a synthetic run (check the `synthetic` flag).
- Figures regenerated after the final training run.
- Plagiarism check on the literature review.
