"""Unit tests: MIME parsing, URL extraction, text cleaning."""
from conftest import FIXTURES

from aimes.connector.parser import parse_file, parse_string
from aimes.textutils import defang, extract_urls, registered_domain, similarity


def test_parses_headers_and_body():
    email = parse_file(FIXTURES / "phishing_sample.eml")
    assert email.sender == "no-reply@micros0ft-verify.tk"
    assert email.sender_name == "Microsoft 365 Support"
    assert email.sender_domain == "micros0ft-verify.tk"
    assert email.reply_to == "account-recovery@mail.ru"
    assert "spf=fail" in email.auth_results
    assert email.received_chain
    assert "verify your identity" in email.clean_text.lower()
    assert "<p>" not in email.clean_text


def test_extracts_href_urls_from_html():
    email = parse_file(FIXTURES / "phishing_sample.eml")
    assert any("secure-verify-account.tk" in url for url in email.urls)


def test_attachment_metadata_without_execution():
    email = parse_file(FIXTURES / "attachment_sample.eml")
    assert len(email.attachments) == 1
    attachment = email.attachments[0]
    assert attachment.filename == "invoice_88231.pdf.exe"
    assert attachment.size > 0
    assert len(attachment.sha256) == 64
    assert not hasattr(attachment, "payload")  # bytes are never retained


def test_body_only_message_flags_missing_headers():
    from aimes.connector.parser import from_plain_text

    email = from_plain_text("just a body with http://example.com/x", "A subject")
    assert email.has_headers is False
    assert email.urls == ["http://example.com/x"]


def test_malformed_message_does_not_raise():
    email = parse_string("not really an email at all\n\nbody")
    assert email.sender == ""
    assert "body" in email.clean_text


def test_url_helpers():
    assert registered_domain("mail.secure-login.co.uk") == "secure-login.co.uk"
    assert registered_domain("http://192.168.1.1/a") == "192.168.1.1"
    assert 0.9 < similarity("micros0ft.com", "microsoft.com") < 1.0
    assert defang("https://a.b/c") == "hxxps://a[.]b/c"
    assert extract_urls("go to www.example.com now") == ["http://www.example.com"]
