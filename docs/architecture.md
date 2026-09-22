# Architecture

## Flow

```
Gmail (OAuth 2.0, read-only)
        |  users.history.list delta polling
        v
  MIME parser  ----> ParsedEmail (headers, text, html, urls, attachment metadata)
        |
        +--> L1 Lexical         TF-IDF + classical classifier      -> score, probs
        +--> L2 Transformer     fine-tuned DistilBERT (ADVANCED)   -> score, probs
        +--> L3 Semantic        embedding vs class centroids       -> score
        +--> L4 Anomaly         Isolation Forest on ham only       -> score
        +--> L5 URL             27 lexical features (+ GBM)        -> score
        +--> L6 Header          SPF/DKIM/DMARC, mismatches         -> score
        +--> L7 Attachment      extension/MIME/macro markers       -> score
        +--> L8 Impersonation   display name vs domain, homoglyphs -> score
        +--> L9 Social eng.     11 pressure categories             -> score
        |
        v
  Risk engine: stacking meta-model over [scores | availability mask],
               weighted fallback, weights renormalised over available layers
        |
        +--> risk 0-100 -> band (SAFE / SUSPICIOUS / HIGH_RISK / CRITICAL)
        +--> base class (HAM / PROMOTIONAL_SPAM / PHISHING)
        +--> threat tags, each with the layer evidence that triggered it
        +--> confidence = agreement x own-confidence x coverage
        |
        v
  SQLite  ->  dashboard (FastAPI + Jinja2, SSE live stream)
          ->  alerts (desktop, Telegram)
          ->  feedback -> gated retraining -> new model version
```

## Why an ensemble rather than one big classifier

A single text classifier sees only the words. It cannot see that the Reply-To
domain differs from the From domain, that the link host is an IP address, or
that the attachment is `invoice.pdf.exe`. Those are structural facts, and each
is cheap to compute and hard to fake simultaneously. The ensemble exists so that
an attacker has to defeat several independent kinds of evidence at once, and so
that each decision can be explained in terms a person can check by hand.

The leave-one-layer-out ablation (experiment E3) is what turns that argument
from a claim into a measurement.

## Abstention

`LayerResult.score` is `Optional[float]`. `None` means "this layer could not run
on this message", which is different from "this layer saw nothing suspicious".
Body-only corpora, missing headers, an untrained model, or an exception inside a
layer all produce abstention. The fusion step divides by the sum of the weights
of the layers that actually ran, so a message analysed by four layers is not
silently penalised relative to one analysed by nine.

## Threading

The monitor runs in a daemon thread with exponential backoff; the dashboard runs
in the main process. SQLite connections are per-thread
(`threading.local`) with WAL enabled, so both can write.

## Extending it

- **Another mailbox type**: implement `connector/base.py:MailProvider` (connect,
  fetch_new, label) and hand it to `Monitor`. Nothing in the detection path
  knows about Gmail.
- **Another layer**: subclass `detectors/base.py:Detector`, return a
  `LayerResult`, register it in `pipeline.py` and add a prior weight in
  `config.DEFAULT_WEIGHTS`. Retraining refits the fusion weights automatically.
- **Another brand to protect**: add a line to `data/brands.txt`.
