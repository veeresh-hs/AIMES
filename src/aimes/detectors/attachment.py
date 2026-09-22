"""Layer 7 - attachment analysis.

Metadata and static markers only.  Attachments are never written to disk and
never executed.  Deeper dynamic analysis would require an isolated sandbox and
is listed as future work rather than simulated here.
"""
from __future__ import annotations

from .base import Detector, clamp01
from ..schema import LayerResult, ParsedEmail

EXECUTABLE_EXTS = {
    "exe", "scr", "com", "pif", "bat", "cmd", "msi", "msp", "cpl", "jar",
    "vb", "vbs", "vbe", "js", "jse", "wsf", "wsh", "ps1", "psm1", "hta",
    "lnk", "reg", "dll", "sys", "app", "apk", "iso", "img", "gadget",
}
MACRO_EXTS = {"docm", "xlsm", "pptm", "xlsb", "dotm", "xltm"}
LEGACY_OFFICE = {"doc", "xls", "ppt", "rtf"}
ARCHIVE_EXTS = {"zip", "rar", "7z", "tar", "gz", "bz2", "ace", "cab", "arj", "z"}
BENIGN_EXTS = {"pdf", "png", "jpg", "jpeg", "gif", "txt", "csv", "docx", "xlsx",
               "pptx", "odt", "ods", "ics", "vcf", "mp3", "mp4", "svg"}
LURE_WORDS = ("invoice", "payment", "receipt", "statement", "remittance",
              "purchase", "order", "shipment", "refund", "salary", "resume",
              "cv", "scan", "urgent", "swift", "bank", "contract")
EXT_MIME = {
    "pdf": ("application/pdf",),
    "zip": ("application/zip", "application/x-zip-compressed", "application/octet-stream"),
    "png": ("image/png",), "jpg": ("image/jpeg",), "jpeg": ("image/jpeg",),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    "xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
}


def extensions(filename: str) -> list[str]:
    return [p.lower() for p in filename.split(".")[1:] if p]


class AttachmentDetector(Detector):
    name = "attachment"

    def _analyze(self, email: ParsedEmail) -> LayerResult:
        if not email.attachments:
            return LayerResult(name=self.name, score=0.0, confidence=0.6,
                               features={"n_attachments": 0},
                               evidence=["No attachments"])

        worst, evidence, per_file = 0.0, [], []
        for att in email.attachments:
            exts = extensions(att.filename)
            ext = exts[-1] if exts else ""
            score, why = 0.0, []

            if ext in EXECUTABLE_EXTS:
                score += 0.70
                why.append(f"Executable file type (.{ext})")
            if len(exts) >= 2 and exts[-2] in BENIGN_EXTS and ext not in BENIGN_EXTS:
                score += 0.25
                why.append(f"Double extension disguises the real type (.{exts[-2]}.{ext})")
            if ext in MACRO_EXTS or att.has_macro_marker:
                score += 0.45
                why.append("Document carries macro content")
            if ext in LEGACY_OFFICE:
                score += 0.15
                why.append(f"Legacy Office format (.{ext}) can carry macros")
            if ext in ARCHIVE_EXTS:
                score += 0.20
                why.append(f"Archive attachment (.{ext}) hides its contents from the scanner")
            if any(w in att.filename.lower() for w in LURE_WORDS) and score > 0:
                score += 0.10
                why.append("File name uses a finance or delivery lure")
            expected = EXT_MIME.get(ext)
            if expected and att.content_type and att.content_type not in expected:
                score += 0.20
                why.append(f"Declared type {att.content_type} does not match .{ext}")
            if att.size == 0:
                score += 0.05
                why.append("Empty attachment")
            if ext == "" :
                score += 0.15
                why.append("Attachment has no file extension")

            score = clamp01(score)
            per_file.append({"filename": att.filename, "ext": ext,
                             "content_type": att.content_type, "size": att.size,
                             "sha256": att.sha256[:16], "risk": round(score, 3),
                             "reasons": why})
            if score > worst:
                worst = score
            if why:
                evidence.append(f"{att.filename}: " + "; ".join(why))

        return LayerResult(
            name=self.name,
            score=worst,
            confidence=0.8,
            features={"n_attachments": len(email.attachments),
                      "max_attachment_risk": round(worst, 3),
                      "files": per_file},
            evidence=evidence or [f"{len(email.attachments)} attachment(s), nothing unusual in the metadata"],
        )
