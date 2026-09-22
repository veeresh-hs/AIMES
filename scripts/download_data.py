"""Fetch the public corpora into data/raw.

Every source here is free and publicly distributed for research.  The licence
of each is printed before anything is downloaded; read them - your report needs
to cite these correctly.  If a mirror is down the script says so and continues;
it never substitutes data from an unknown source.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from aimes.config import settings
from aimes.logging_conf import get_logger

log = get_logger("download_data")

SOURCES = {
    "spamassassin": {
        "licence": "Apache SpamAssassin public corpus - free for research use.",
        "cite": "Apache SpamAssassin Project. Public Corpus. https://spamassassin.apache.org/old/publiccorpus/",
        "files": [
            "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
            "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
            "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
            "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
            "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
        ],
    },
    "enron_spam": {
        "licence": "Enron-Spam (Metsis, Androutsopoulos & Paliouras, CEAS 2006) - public research corpus.",
        "cite": "V. Metsis, I. Androutsopoulos, G. Paliouras. Spam Filtering with Naive Bayes - Which Naive Bayes? CEAS 2006.",
        "files": [f"https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{i}.tar.gz"
                  for i in range(1, 7)],
    },
}

MANUAL = {
    "nazario": ("Nazario phishing corpus - request/download from "
                "https://monkey.org/~jose/phishing/ and put the .mbox files in "
                "data/raw/nazario/. Full headers, so it exercises the header layer."),
    "phishing_csv": ("Any CC-licensed phishing email CSV (for example the curated "
                     "phishing email dataset on Zenodo) with text and label columns, "
                     "placed in data/raw/phishing_csv/."),
    "url_dataset": ("A phishing-URL CSV such as PhiUSIIL from the UCI ML Repository "
                    "(CC BY 4.0), placed in data/raw/url_dataset/. Trains layer 5 only."),
}


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        log.info("already have %s", dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AIMES-student-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as fh:
            shutil.copyfileobj(response, fh)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.error("could not download %s (%s)", url, exc)
        if dest.exists():
            dest.unlink()
        return False


def extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffixes[-2:] in ([".tar", ".bz2"], [".tar", ".gz"]) or archive.suffix in (".bz2", ".gz"):
            with tarfile.open(archive) as tar:
                members = [m for m in tar.getmembers()
                           if not (m.name.startswith("/") or ".." in m.name)]
                try:
                    tar.extractall(target, members=members, filter="data")
                except TypeError:  # python < 3.12
                    tar.extractall(target, members=members)
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
    except Exception as exc:
        log.error("could not extract %s: %s", archive.name, exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES),
                    help="subset of: " + " ".join(SOURCES))
    args = ap.parse_args()

    raw = Path(settings.data_dir) / "raw"
    print("\nLicences of the corpora this script downloads:\n")
    for name, source in SOURCES.items():
        print(f"  {name}: {source['licence']}\n    cite as: {source['cite']}")
    print()

    for name in args.only:
        source = SOURCES.get(name)
        if not source:
            log.warning("unknown source %s", name)
            continue
        target = raw / name
        cache = raw / "_archives"
        for url in source["files"]:
            archive = cache / url.rsplit("/", 1)[-1]
            if download(url, archive):
                extract(archive, target)

        if name == "enron_spam":  # normalise enron1/ham -> enron_spam/ham
            for kind in ("ham", "spam"):
                merged = target / kind
                merged.mkdir(parents=True, exist_ok=True)
                for path in target.glob(f"enron*/{kind}/*.txt"):
                    destination = merged / f"{path.parent.parent.name}_{path.name}"
                    if not destination.exists():
                        try:
                            path.replace(destination)
                        except OSError:
                            pass

    print("\nCorpora that must be fetched by hand (licence or registration):")
    for name, note in MANUAL.items():
        print(f"  - {name}: {note}")
    print("\nWhen you are done:  python scripts/train_all.py")


if __name__ == "__main__":
    main()
