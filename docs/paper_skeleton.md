# Research paper skeleton (6-8 pages, IEEE two-column)

Working title: **Layer Abstention and Evidence Fusion for Explainable Phishing
Detection in Personal Mailboxes**

Fill the bracketed placeholders from `data/reports/experiments.json` after a run
on real corpora. Do not write a number you have not measured.

## Abstract (150-200 words)
Problem; nine-layer approach with explicit abstention; fusion by logistic
stacking; evaluation on [corpora] with [N] messages; headline result
[ROC-AUC, FPR at 95% recall]; cross-corpus result [value]; and the ablation
finding that [layer] contributes most.

## I. Introduction
Phishing losses; why content-only filters miss structural evidence; why
explainability matters for a non-expert recipient. Contributions:
1. a nine-layer detector with a principled abstention mechanism,
2. a fusion rule that renormalises over available evidence,
3. an ablation quantifying each layer's contribution,
4. a cross-corpus evaluation and a cost-sensitive operating point analysis.

## II. Related work
Content filtering; URL-based detection; authentication-based heuristics;
transformers for phishing; ensembles; explainability. Close with the gap:
single-signal, in-corpus evaluation without false-positive cost.

## III. Methodology
A. Message representation. B. The nine layers (compress Section 6 of the report;
one paragraph each, one equation where it earns its place). C. Abstention:
define the availability mask m_i and the fused score

    s = sum_i (w_i m_i s_i) / sum_i (w_i m_i)

and the stacking variant over [scores | mask]. D. Risk bands and tags.

## IV. Experimental setup
Corpora and licences; de-duplication before splitting; split strategy; hardware
(CPU only, [model], [RAM]); metrics, with the argument for PR-AUC and FPR at
fixed recall over plain accuracy under class imbalance.

## V. Results
Table I single layers. Table II fusion vs best single layer. Table III ablation.
Table IV cross-corpus. Figure 1 ROC/PR curves. Figure 2 cost sweep.
Table V latency.

## VI. Discussion
Which evidence types are complementary rather than redundant; where abstention
changes the outcome; the in-corpus versus cross-corpus gap; false-positive cost
in a personal mailbox.

## VII. Threats to validity
Corpus age and provenance; label noise; the tag layer is rule-derived, not
supervised; single language; no live A/B deployment.

## VIII. Conclusion and future work

## References (IEEE)
Include: Metsis et al. CEAS 2006 (Enron-Spam); Apache SpamAssassin public corpus;
Nazario phishing corpus; Pedregosa et al. 2011 (scikit-learn); Lundberg & Lee
2017 (SHAP); Liu et al. 2008 (Isolation Forest); Sanh et al. 2019 (DistilBERT);
Reimers & Gurevych 2019 (Sentence-BERT).

## Venue suggestions
Student track at an IEEE/ACM regional conference, or a workshop on ML for
security. Check the page limit and template before formatting.
