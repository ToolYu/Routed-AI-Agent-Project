import argparse
import base64
import os
from email.message import EmailMessage
from typing import List

# Google OAuth / Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _load_env_file(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


def get_gmail_service(
    client_secrets_file: str | None = None,
    token_file: str | None = None,
):
    _load_env_file()
    client_secrets_file = client_secrets_file or os.environ.get("GOOGLE_CLIENT_SECRETS", "google_client_secret.json")
    token_file = token_file or os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

    if not os.path.exists(client_secrets_file):
        raise SystemExit(
            "Google OAuth client secrets not found: "
            f"{client_secrets_file}\n"
            "- Specify the file via --client-secrets / set GOOGLE_CLIENT_SECRETS in .env\n"
            "- To obtain it: Google Cloud Console → APIs & Services → Credentials → Create Credentials → OAuth client ID (Desktop) → Download JSON"
        )

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            # Try to open a local server (best UX). If fails, fall back to console.
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                creds = flow.run_console()
        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


def _parse_recipients(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _attach_files(msg: EmailMessage, paths: List[str]):
    import mimetypes

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


def send_via_gmail_api(
    service,
    from_email: str,
    to_emails: List[str],
    subject: str,
    body: str,
    attachments: List[str] | None = None,
):
    if not to_emails:
        raise ValueError("No recipients provided")

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = ", ".join(to_emails)
    msg["Subject"] = subject
    msg.set_content(body)

    _attach_files(msg, attachments or [])

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()


def main():
    _load_env_file()
    parser = argparse.ArgumentParser(description="Send email via Gmail API with OAuth (persisted consent)")
    parser.add_argument("--to", required=True, help="Recipient emails, comma/semicolon separated")
    parser.add_argument("--subject", default=os.environ.get("SUBJECT", ""))
    parser.add_argument("--body", default=os.environ.get("BODY", ""))
    parser.add_argument("--body-file", help="Read email body from a file (overrides --body)")
    parser.add_argument("--from-email", default=os.environ.get("FROM_EMAIL", ""), help="From email (Gmail address)")
    parser.add_argument("--attach", action="append", default=[], help="Attachment path (repeatable)")
    parser.add_argument("--client-secrets", default=os.environ.get("GOOGLE_CLIENT_SECRETS", "google_client_secret.json"))
    parser.add_argument("--token-file", default=os.environ.get("GOOGLE_TOKEN_FILE", "token.json"))

    args = parser.parse_args()

    body_text = args.body
    if args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            body_text = f.read()

    to_emails = _parse_recipients(args.to)
    from_email = args.from_email
    if not from_email:
        # If omitted, Gmail API will use the authorized account. You can still set it explicitly.
        from_email = ""

    service = get_gmail_service(client_secrets_file=args.client_secrets, token_file=args.token_file)
    resp = send_via_gmail_api(
        service=service,
        from_email=from_email or "me",
        to_emails=to_emails,
        subject=args.subject,
        body=body_text,
        attachments=args.attach,
    )
    msg_id = resp.get("id")
    thread_id = resp.get("threadId")
    print(f"Sent. messageId={msg_id}, threadId={thread_id}")


if __name__ == "__main__":
    main()
