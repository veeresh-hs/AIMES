"""Unit tests for the rule/feature layers that run without a trained model."""
from conftest import FIXTURES

from aimes.connector.parser import from_plain_text, parse_file
from aimes.detectors.attachment import AttachmentDetector
from aimes.detectors.header import HeaderDetector
from aimes.detectors.impersonation import ImpersonationDetector
from aimes.detectors.social_engineering import SocialEngineeringDetector
from aimes.detectors.url import URLDetector, url_features


def test_url_features_detect_ip_and_at():
    features = url_features("http://user@192.168.1.9/login/verify?x=1")
    assert features["is_ip"] == 1.0
    assert features["has_at"] == 1.0
    assert features["sensitive_word_hits"] >= 2


def test_url_layer_ranks_phishing_above_normal():
    detector = URLDetector()
    bad = detector.analyze(from_plain_text(
        "http://paypal.com.secure-verify.tk/login/confirm?u=%41%42%43%44"))
    good = detector.analyze(from_plain_text("https://github.com/torvalds/linux"))
    assert bad.score > good.score
    assert "hxxp" in bad.features["top_url_defanged"]


def test_url_layer_reports_zero_when_no_links():
    result = URLDetector().analyze(from_plain_text("no links here at all"))
    assert result.score == 0.0


def test_header_layer_flags_auth_failure_and_mismatch():
    result = HeaderDetector().analyze(parse_file(FIXTURES / "phishing_sample.eml"))
    assert result.features["spf"] == "fail"
    assert result.features["reply_to_mismatch"] == 1
    assert result.score > 0.5


def test_header_layer_quiet_on_legitimate_mail():
    result = HeaderDetector().analyze(parse_file(FIXTURES / "legit_sample.eml"))
    assert result.features["spf"] == "pass"
    assert result.score < 0.3


def test_header_layer_abstains_without_headers():
    result = HeaderDetector().analyze(from_plain_text("body only"))
    assert result.score is None and result.available is False


def test_attachment_layer_catches_double_extension():
    result = AttachmentDetector().analyze(parse_file(FIXTURES / "attachment_sample.eml"))
    assert result.score > 0.8
    assert any("xecutable" in reason or "ouble extension" in reason
               for reason in result.evidence)


def test_impersonation_layer_detects_lookalike_domain():
    result = ImpersonationDetector().analyze(parse_file(FIXTURES / "phishing_sample.eml"))
    assert result.score > 0.5
    assert result.features["brand_claimed"] == "microsoft"


def test_impersonation_quiet_on_normal_sender():
    result = ImpersonationDetector().analyze(parse_file(FIXTURES / "legit_sample.eml"))
    assert result.score < 0.3


def test_social_engineering_scores_pressure_language():
    pressure = SocialEngineeringDetector().analyze(from_plain_text(
        "Urgent: verify your account immediately or it will be suspended. "
        "Send the OTP now to avoid legal action."))
    calm = SocialEngineeringDetector().analyze(from_plain_text(
        "Notes from the meeting are attached, nothing urgent on my side."))
    assert pressure.score > calm.score
    assert pressure.features["urgency"] >= 1


def test_layer_never_raises_on_broken_input():
    class Broken(URLDetector):
        def _analyze(self, email):
            raise ValueError("boom")

    result = Broken().analyze(from_plain_text("x"))
    assert result.score is None and result.available is False and "boom" in result.error
