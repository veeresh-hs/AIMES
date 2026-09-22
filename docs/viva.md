# Viva preparation

Twenty questions examiners actually ask, with answers grounded in this codebase.

**1. Why nine layers instead of one good classifier?**
A text classifier only sees words. It cannot see that Reply-To points to a
different domain, that the link host is a bare IP, or that the attachment is
`invoice.pdf.exe`. Experiment E3 (leave-one-layer-out) measures what each layer
adds rather than assuming it adds something.

**2. What happens when a layer cannot run?**
It returns `score=None` and is excluded. The fusion divides by the sum of the
weights of the layers that did run. A neutral 0.5 would be a fabricated
observation; see `test_missing_layers_do_not_count_as_zero`.

**3. Where do the fusion weights come from?**
Fitted, not chosen. `ml/training.py:fit_fusion` fits a logistic regression on the
validation split - where the base layers are already frozen - and the weighted
variant uses the normalised non-negative coefficients. The priors in `config.py`
are only used before the first training run.

**4. Why fit the fusion on validation rather than in-fold stacking?**
Because the base layers must not score their own training data. Blending on a
held-out validation split gives unbiased inputs to the meta-learner with far less
complexity than nested cross-validation. The test split is still untouched.

**5. Why three classes rather than the ten threat types you display?**
No public corpus labels "impersonation" or "CEO fraud". Training a ten-class
model would need invented labels. So three classes are learned from real labels,
and the finer types are emitted as tags derived from layer evidence, with the
trigger recorded for each.

**6. How do you avoid data leakage?**
De-duplication happens before splitting (`preprocessing.py`); near-duplicate
messages are common in these corpora. Vectorisers and models see the train split
only, fusion sees validation, and the test split is used once.

**7. Your accuracy on SpamAssassin is high. Is that impressive?**
No. That corpus is from 2003 and is easy. The number that matters is E4, the
cross-corpus test: train on SpamAssassin plus Enron-Spam, test on an unseen
phishing corpus. It is lower, and that is the honest measure of generalisation.

**8. Why not accuracy as the headline metric?**
Class imbalance. With 90% ham, a model that flags nothing scores 90%. The system
reports PR-AUC, and the false-positive rate at fixed recall (E5), because in a
personal mailbox a missed threat and a misfiled invoice have very different costs.

**9. How did you choose the risk band thresholds?**
Defaults are in `.env`, but E5 sweeps thresholds under a stated cost model
(five false negatives weighted against one false positive) and reports the
cost-optimal point. The ratio is a parameter to be justified, not a fact.

**10. Does the system delete or move mail?**
No. The OAuth scope is `gmail.readonly` by default. Trash and delete scopes are
never requested, so the application cannot delete mail even if compromised.
Adding an `AIMES/<BAND>` label is opt-in and requires two settings to be enabled.

**11. Do you fetch the suspicious URLs to check them?**
Never. That would confirm delivery to the attacker and could execute a drive-by.
All URL analysis is lexical, on the string. Links are displayed defanged.

**12. How do you handle malicious attachments safely?**
Metadata and static markers only: extension, double extension, MIME mismatch, and
macro markers (via oletools when installed). Bytes are hashed in memory and
dropped - never written to disk, never opened. Real detonation needs a sandbox,
which is listed as future work rather than simulated.

**13. Could a user poison your model by giving false feedback?**
That is why feedback is not applied online. Corrections accumulate, a minimum
batch triggers a candidate retrain, and the promotion gate compares the candidate
against the incumbent. If macro-F1 does not improve, or the false-positive rate
worsens, the candidate is not activated.

**14. What is the promotion gate exactly?**
`registry.should_promote`: +0.002 macro-F1 required and no more than +0.005 FPR.
A model that catches more phishing by flagging good mail does not ship.

**15. How is the transformer layer handled on a laptop without a GPU?**
It is off in FAST mode and reports itself unavailable with a reason, rather than
pretending to have run. Mode is chosen from available RAM unless pinned in `.env`.

**16. What does "confidence" mean in your output?**
A blend of three things: agreement between layers (exponential in their standard
deviation), the layers' own confidences, and coverage - how many layers were able
to run. Low agreement lowers confidence even when the score is high.

**17. Is SHAP being used correctly?**
It is applied to the fusion meta-model and the linear lexical model, where the
attribution is meaningful. Transformer attention weights are deliberately not
presented as explanations, because attention is not causal attribution.

**18. What is the biggest weakness of this project?**
Corpus age and provenance. Public phishing corpora lag current campaigns, so
absolute numbers overstate performance on today's mail. Second: the brand list
for impersonation is small and hand-maintained.

**19. How would you deploy this for a real organisation?**
Not as-is. It would need a mail-gateway integration rather than polling, an
isolated attachment sandbox, per-user threshold calibration, monitoring for
drift, and an appeal path for false positives. The dashboard currently binds to
localhost and would need real authentication behind it.

**20. What did you build yourself versus what came from a library?**
Libraries supply the learning algorithms (scikit-learn), HTML parsing and the web
framework. The parsing hardening, the nine layers and their features, the
abstention mechanism, the fusion and risk engine, the explanation ranking, the
registry and promotion gate, the experiment suite and the dashboard are project
code. Be ready to open any file and walk through it.

## Likely live demo requests
- Analyse `tests/fixtures/phishing_sample.eml` from the command line and explain
  every layer score aloud.
- Open an analysed email in the dashboard and read the contribution table.
- Show `models/registry.json` and explain why a version was or was not promoted.
- Edit `data/brands.txt`, add your college domain, and re-run to show the
  impersonation layer picking it up without a code change.
