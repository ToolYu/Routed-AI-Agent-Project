import os
import ssl
import sys
import smtplib
import argparse
import getpass
import mimetypes
from email.message import EmailMessage
from email.policy import SMTP as SMTPPolicy
import html as html_mod


def _load_env_file(path: str = ".env") -> None:
    """Lightweight .env loader (KEY=VALUE, ignores comments/blank lines)."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail silently if .env cannot be read; environment variables may still be set externally.
        pass


def _parse_recipients(raw: str) -> list:
    if not raw:
        return []
    # Support comma or semicolon separated lists
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _attach_files(msg: EmailMessage, paths: list[str]) -> None:
    for p in paths or []:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Attachment not found: {p}")
        ctype, encoding = mimetypes.guess_type(p)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(p, "rb") as fp:
            data = fp.read()
        filename = os.path.basename(p)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)


def _to_html(text: str) -> str:
    text = text or ""
    paras = [p.strip() for p in text.split("\n\n")]
    parts = []
    for p in paras:
        if not p:
            continue
        safe = html_mod.escape(p).replace("\n", "<br>")
        parts.append(f"<p>{safe}</p>")
    return "\n".join(parts) or f"<p>{html_mod.escape(text).replace('\n','<br>')}</p>"


def _make_ssl_context() -> ssl.SSLContext:
    # Allow custom CA bundle via env vars (helpful behind corporate proxies)
    for var in ("SMTP_CA_FILE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        p = os.environ.get(var)
        if p and os.path.isfile(p):
            try:
                ctx = ssl.create_default_context(cafile=p)
                if os.environ.get("DEBUG_SSL"):
                    print(f"Using CA bundle from {var}={p}", file=sys.stderr)
                return ctx
            except Exception:
                pass

    try:
        import certifi  # type: ignore
        ctx = ssl.create_default_context(cafile=certifi.where())
        if os.environ.get("DEBUG_SSL"):
            print(f"Using certifi CA bundle: {certifi.where()}", file=sys.stderr)
        return ctx
    except Exception:
        if os.environ.get("DEBUG_SSL"):
            print("Using system default CA bundle", file=sys.stderr)
        return ssl.create_default_context()


def send_email(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    from_email: str,
    to_emails: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
    use_starttls: bool = False,
):
    if not to_emails:
        raise ValueError("No recipients provided")

    policy = SMTPPolicy.clone(max_line_length=998)
    msg = EmailMessage(policy=policy)
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    msg.set_content(body or "", subtype="plain", charset="utf-8", cte="8bit")
    # Add HTML alternative to improve rendering across clients
    try:
        msg.add_alternative(_to_html(body), subtype="html", charset="utf-8")
    except Exception:
        pass

    _attach_files(msg, attachments or [])

    context = _make_ssl_context()
    if use_starttls:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)


def main():
    _load_env_file()  # Load from .env if present (non-fatal)

    parser = argparse.ArgumentParser(description="Send an email via SMTP (supports attachments)")
    parser.add_argument("--to", help="Recipient emails (comma or semicolon separated)")
    parser.add_argument("--subject", default=os.environ.get("SUBJECT", ""), help="Email subject")
    parser.add_argument("--body", default=os.environ.get("BODY", ""), help="Email body text")
    parser.add_argument("--body-file", help="Read email body from a file (overrides --body if provided)")
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Path to attachment (use multiple --attach for several files)",
    )

    parser.add_argument("--from-email", default=os.environ.get("FROM_EMAIL"), help="From email address")
    parser.add_argument("--username", default=os.environ.get("SMTP_USERNAME"), help="SMTP username (often same as from email)")
    parser.add_argument("--password", default=os.environ.get("SMTP_PASSWORD"), help="SMTP password or app-specific password")

    parser.add_argument("--smtp-server", default=os.environ.get("SMTP_SERVER", "smtp.gmail.com"), help="SMTP server hostname (e.g., smtp.gmail.com)")
    parser.add_argument("--smtp-port", type=int, default=int(os.environ.get("SMTP_PORT", "465")), help="SMTP port (465 SSL, 587 STARTTLS)")
    parser.add_argument(
        "--starttls",
        action="store_true",
        help="Use STARTTLS (typically with port 587). Default uses SSL (port 465)",
    )

    args = parser.parse_args()

    to_emails = _parse_recipients(args.to)

    body_text = args.body
    if args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            body_text = f.read()

    username = args.username or os.environ.get("SMTP_USERNAME")
    from_email = args.from_email or os.environ.get("FROM_EMAIL") or username
    password = args.password or os.environ.get("SMTP_PASSWORD")

    if not username:
        raise SystemExit("Missing SMTP username (use --username or set SMTP_USERNAME)")
    if not from_email:
        raise SystemExit("Missing from email (use --from-email or set FROM_EMAIL)")
    if not password:
        # Prompt if not provided
        password = getpass.getpass("SMTP password (or app password): ")

    send_email(
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        username=username,
        password=password,
        from_email=from_email,
        to_emails=to_emails,
        subject=args.subject,
        body=body_text,
        attachments=args.attach,
        use_starttls=args.starttls,
    )


if __name__ == "__main__":
    main()
