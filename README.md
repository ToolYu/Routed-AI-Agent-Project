# Routed AI Outreach Agent

CLI-oriented workflow for generating and optionally sending personalized outreach messages using LangGraph and LangChain. The agent ingests sample profiles, produces persona-aware copy, and can deliver the result through Gmail OAuth for now (will explore SMTP option).

## Project Layout
- `agents/` — agent pipelines, email utilities, OAuth helper, sample CSV.
- `datasets/clustering/` — sample sender/target profiles (`user_features.csv`) and clustering outputs.
- `notebooks/` — exploratory notebooks (EDA, clustering, feature prep).
- `reports/profiling/` — HTML profiling reports.
- `requirements.txt` — Python dependencies.

## Prerequisites
- Python 3.10+.
- Groq or OpenAI API key (defaults to Groq).
- Optional: Gmail account for sending emails (OAuth preferred).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy the sample env file and fill in secrets:
```bash
cp .env.example .env
```

Minimum `.env` entries:
```
GROQ_API_KEY=sk_your_key
LLM_PROVIDER=groq
GROQ_MODEL_VERSION=llama-3.3-70b-versatile

GOOGLE_CLIENT_SECRETS=google_client_secret.json
GOOGLE_TOKEN_FILE=token.json
```
Add `SEND_TO`, `INTENT_TO_CONNECT`, or SMTP credentials if desired.

## Gmail OAuth (Optional Send Step)
1. In Google Cloud Console enable **Gmail API**, configure the OAuth consent screen (External, add yourself as a Test user), and create a **Desktop App** OAuth client.
2. Download the JSON, place it in project root (rename to `google_client_secret.json` or update `GOOGLE_CLIENT_SECRETS` path).
3. First run triggers a browser flow; approve the app ("Advanced → Continue") and a `token.json` file is created for future runs.

## Running the Agent
The sample CLI expects to run inside `agents/` so paths resolve correctly.
```bash
source .venv/bin/activate
cd agents
python langgraph_outreach_pipeline.py
```

Workflow:
1. Prompts for an outreach intent (press Enter to use `INTENT_TO_CONNECT` from `.env`).
2. Loads sender/target profiles from `agents/user_features.csv` (same as `datasets/clustering/user_features.csv`).
3. LangGraph executes:
   - **Channel choice** → defaults to email.
   - **PersonaAgent** → distills overlaps, tone, slot map.
   - **ContentAgent** → drafts channel-ready copy with guardrails.
   - **Store plan / Confirm** → records final messages.
4. Saves drafts to `agents/personalized_messages.csv`.
5. CLI previews the first email and can send via Gmail OAuth if configured.

## Optional Email Tools
- `python agents/send_email.py --to someone@example.com --body "..."` (SMTP + app password).
- `python agents/gmail_oauth_sender.py --to someone@example.com --body-file agents/personalized_messages.csv` (Gmail API).

## Development Tips
- Ignore local artifacts (PDFs, large CSVs) via `.gitignore`.
- If using another LLM provider, set `LLM_PROVIDER=openai` (or `azure`) and supply the matching API keys/env vars.
- For notebooks, activate the venv before launching Jupyter to pick up dependencies.
