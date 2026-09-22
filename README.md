# AIMES - AI-based multi-layer spam and email threat detection

A local, free, open-source email threat detection system. Nine detection layers
analyse each message, an ensemble fuses their evidence into a 0-100 risk score,
and a dashboard explains exactly why each score was given.

Built as a B.Tech CSE (AI/ML + Cybersecurity) final-year project. No paid APIs,
no cloud services, no GPU required. Everything runs on your own machine.

---

## What it actually does

1. Connects to Gmail through OAuth 2.0 (read-only by default - it is structurally
   incapable of deleting your mail, and never asks for your password).
2. Parses each new message: headers, body, HTML, links, attachment metadata.
3. Runs nine layers over it:

   | # | Layer | What it uses | Trained? |
   |---|---|---|---|
   | 1 | Lexical | TF-IDF (word 1-2gram + char 3-5gram), best of NB / LogReg / LinearSVC / RandomForest | yes |
   | 2 | Transformer | fine-tuned DistilBERT, ADVANCED mode only | optional |
   | 3 | Semantic | MiniLM embeddings vs class centroids, LSA fallback offline | yes |
   | 4 | Anomaly | Isolation Forest fitted on legitimate mail only | yes |
   | 5 | URL | 27 lexical URL features, gradient boosting when a URL corpus is present | optional |
   | 6 | Header | SPF / DKIM / DMARC results, Reply-To and Return-Path mismatch, relay chain | rules |
   | 7 | Attachment | extension, double extension, MIME mismatch, static macro markers | rules |
   | 8 | Impersonation | display-name vs domain, homoglyphs, edit distance to known brands | rules |
   | 9 | Social engineering | 11 categories of pressure language as normalised densities | rules |

4. Fuses the layers (logistic stacking, with a transparent weighted fallback),
   producing a risk score, a class (HAM / PROMOTIONAL_SPAM / PHISHING), threat
   tags, and a confidence value.
5. Alerts you above a configurable threshold: desktop notification, dashboard
   stream, and optionally a free Telegram bot.
6. Records your corrections and uses them in a gated retraining cycle.

### Two design decisions worth knowing before you read the code

**Labels are two-tier.** No public corpus labels "impersonation" or "CEO fraud",
so a flat 10-class classifier would need invented labels. Instead the supervised
model predicts three classes that real corpora actually contain, and the finer
threat types are emitted as *tags* derived from layer evidence, each with the
trigger recorded. That is defensible in a viva; a model trained on made-up labels
is not.

**A layer that cannot run abstains.** Body-only corpora have no headers, so the
header layer returns `None` rather than a neutral 0.5. The ensemble drops it and
renormalises the remaining weights. Nothing is ever substituted for missing
evidence.

---

## Quick start (Windows 10/11)

Prerequisites: Python 3.11 64-bit (tick "Add python.exe to PATH" during install),
about 12 GB free disk if you download every corpus, 8 GB RAM for ADVANCED mode.

```bat
cd C:\path\to\aimes
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
python -c "import secrets;print(secrets.token_hex(32))"
REM paste that value into AIMES_SESSION_SECRET in .env

python scripts\init_db.py --user admin
python scripts\download_data.py
python scripts\train_all.py
python scripts\run_dashboard.py
```

Then open http://127.0.0.1:8000 and sign in.

On macOS or Linux the only difference is `source venv/bin/activate` and forward
slashes in paths.

### Try it without a mailbox

```bat
python scripts\analyze_eml.py tests\fixtures\phishing_sample.eml
python scripts\analyze_eml.py tests\fixtures\legit_sample.eml
```

Or sign in to the dashboard, go to Settings, and upload any `.eml` file.

### Connect Gmail

1. Google Cloud Console -> create a project -> enable the Gmail API.
2. OAuth consent screen -> External -> add your own address as a test user.
3. Credentials -> Create credentials -> OAuth client ID -> **Desktop app** ->
   download the JSON -> save it as `secrets\credentials.json`.
4. `python scripts\connect_gmail.py` - your browser opens for consent.
5. `python scripts\connect_gmail.py --watch` to keep polling, or use the
   Start monitoring button on the dashboard.

The token is encrypted at rest if you set `AIMES_TOKEN_KEY` and install
`cryptography`; otherwise the app warns you loudly that it is storing a
refresh token in plaintext.

---

## Running the experiments

```bat
python scripts\train_all.py
python scripts\run_experiments.py
```

Writes `data\reports\experiments.json`, `experiments.md` and figures:

- **E1** every layer on its own, same held-out split
- **E1b** the four lexical baselines against each other
- **E2** weighted fusion vs logistic stacking vs the best single layer
- **E3** leave-one-layer-out ablation across all nine layers
- **E4** cross-corpus: train on SpamAssassin + Enron-Spam, test on an unseen
  phishing corpus - the number that tells you whether it generalises
- **E5** FPR at 90/95/99% recall, plus a cost-sensitive threshold sweep
- **E6** per-layer latency, mean and p95

Those tables go straight into your report. Do not copy numbers by hand.

### Data

`scripts\download_data.py` fetches the SpamAssassin public corpus and Enron-Spam
and prints the licence and citation for each. Three more corpora need a manual
download (licence or registration) and the script tells you exactly where each
one goes:

```
data/raw/spamassassin/{easy_ham,hard_ham,spam,spam_2}/   full RFC-822 messages
data/raw/enron_spam/{ham,spam}/*.txt                     body-only
data/raw/nazario/*.mbox                                  phishing, full headers
data/raw/phishing_csv/*.csv                              any text+label CSV
data/raw/url_dataset/*.csv                               URL+label, layer 5 only
```

If nothing is present, training falls back to a **synthetic generator** so you
can verify the plumbing offline. Synthetic runs stamp every report with a
warning. Synthetic numbers are not results and must never appear in your report.

---

## How it is put together

```
src/aimes/
  config.py          settings from .env, runtime mode, default weights
  schema.py          ParsedEmail, LayerResult, AnalysisResult
  textutils.py       HTML stripping, URL extraction, domain and distance helpers
  connector/         MIME parser, Gmail OAuth, Gmail client, token store
  detectors/         the nine layers, each returning a LayerResult
  ensemble/          fusion, risk bands
  explain/           SHAP contributions, human-readable evidence
  ml/                datasets, preprocessing, training, evaluation, registry
  alerts/            desktop, Telegram, dashboard stream
  api/               FastAPI app and signed-cookie sessions
  web/               templates and the dashboard stylesheet
  database.py        SQLite persistence (stdlib sqlite3, no ORM)
  pipeline.py        loads the active model version and runs every layer
  monitor.py         polling loop with backoff
scripts/             init_db, download_data, train_all, run_experiments,
                     connect_gmail, analyze_eml, retrain_from_feedback, run_dashboard
tests/               unit + integration tests and .eml fixtures
docs/                report outline, paper skeleton, viva preparation
```

Persistence is plain `sqlite3` from the standard library rather than an ORM:
fewer dependencies to install on a lab machine, and the SQL is visible in
`database.py` for your report.

### Model versioning

Each training run writes `models/v<N>/` and an entry in `models/registry.json`.
A new version becomes active only if it clears the promotion gate: macro-F1 must
improve by at least 0.002 **and** the false-positive rate must not rise by more
than 0.005. A model that catches more phishing by flagging good mail does not
ship. The gate lives in `aimes/ml/registry.py`.

### Feedback learning

Marking a verdict wrong in the dashboard records a correction. Nothing changes
immediately. `scripts\retrain_from_feedback.py` collects at least 20 corrections,
appends them as a CSV corpus, retrains, and lets the promotion gate decide. This
is deliberate: immediate online updating from user clicks is a poisoning vector.

Corrections can only be reused if `AIMES_STORE_RAW_BODY=true`, because otherwise
the text is not retained.

---

## Tests

```bat
pytest -q
```

36 unit and integration tests cover parsing, each rule layer, fusion arithmetic
(including that abstaining layers are not treated as zeros), the promotion gate,
database round-trips and the end-to-end pipeline. `tests\test_api.py` skips
itself if FastAPI is not installed.

---

## Privacy and safety

- Read-only Gmail scope by default; the delete and trash scopes are never requested.
- Message bodies are not stored unless you opt in, and are purged after a
  configurable retention period.
- Sender addresses are hashed before they reach the log files.
- Links are never fetched, resolved or previewed - URL analysis is purely lexical.
- Attachments are hashed in memory and dropped; nothing is written to disk or
  executed. Macro detection is static.
- The dashboard binds to 127.0.0.1 and requires a login. Do not expose it to a
  network without putting real authentication in front of it.
- Detection is an aid, not a guarantee. Treat the score as evidence, not proof.

## Known limits

- The transformer layer needs ~2 GB of downloads and is off in FAST mode.
- No sandboxed attachment detonation - that needs isolation this project does not
  provide, and is listed as future work rather than faked.
- Cross-corpus results are the honest ones. In-corpus accuracy on SpamAssassin is
  always flattering because spam from 2003 is easy.
- The impersonation brand list is small and editable (`data/brands.txt`); it is a
  heuristic layer, not a brand-protection product.

## Licence and citations

The code is yours to submit. The corpora are not: cite SpamAssassin, Enron-Spam
(Metsis et al., CEAS 2006), and the Nazario phishing corpus in your report, along
with scikit-learn and any dataset you add. `scripts\download_data.py` prints the
citation for each source it touches.
