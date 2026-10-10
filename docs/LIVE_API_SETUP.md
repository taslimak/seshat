# Connecting real provider APIs safely

The repository contains provider SDK adapters for OpenAI, Anthropic and Gemini.
The workflow `.github/workflows/manual-live-api.yml` executes a **manual**
pilot only, not the twice-daily background job.

## Set up credentials

Use **Settings → Secrets and variables → Actions → New repository secret**
in GitHub. Create only the provider key(s) you plan to use:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and/or `GEMINI_API_KEY`.
Also set `SESHAT_PROMPT_HASH_KEY` to a strong random secret. Never paste these
values in a commit, PR comment, CLI argument, model name, log, or workflow input.

These are secret **names**, not secret values. The code contains no API keys.
The provider SDK reads the selected key from the runtime environment. The
manual workflow checks presence without printing the value.

Personal ChatGPT/Claude subscriptions do not automatically include API billing.
Configure an API project/account with billing and provider-side spending limits.

## Run first live pilot

1. Merge the live-pilot PR after CI checks pass.
2. Navigate to **Actions → Manual paid API smoke test → Run workflow**.
3. Choose a provider, enter an exact verified API model ID from
   `configs/model_catalog.json`, and explicitly approve paid calls.
4. The workflow generates an isolated experiment config with at most
   **2 requests**, each capped at **128 output tokens**, zero retries and a
   **USD 0.20 price-table estimate cap**.
5. The dry run runs first. The real collector requires `--execute-live` and
   `--max-cost-usd 0.20`. The Results artifact is retained for 7 days.

This dollar cap uses estimated list prices, not a guaranteed invoice amount.
Enforce additional provider-side limits. Failed or ambiguous network requests
may still be billed. Unknown models or unverified capabilities are rejected.

## Data and privacy

The artifact may contain experimental prompt content and provider-derived
metadata. Treat it as sensitive and control Actions/artifact access. GitHub
Actions artifacts are temporary, **not durable dataset storage**.
No API secret values are saved into experiment configs or tracked files.

For automatic twice-daily *paid* collection, first provision durable dataset
storage and a persistent shared spending ledger; the existing scheduled
workflow continues using mock providers only.
