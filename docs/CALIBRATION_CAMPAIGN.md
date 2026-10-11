# Real-label calibration collection campaign

The old two-call manual smoke test is retained for connectivity verification.
The new `manual-calibration-batch.yml` is a separate explicitly approved
collection campaign. Each invocation generates 600 candidate prompts and rotates variants using
the GitHub run ID, samples up to 180 requests, allows up to 512 output tokens
per response, disables retries, and limits estimated spend to **$2.00/run**.
Actual provider bills can differ from estimates.

## Before triggering

1. Merge this PR; CI must pass.
2. Ensure the matching API key and `SESHAT_PROMPT_HASH_KEY` are Actions
   Secrets. Select a model supported by your API account and verified in the
   catalog; check pricing before executing.
3. In Actions → Manual calibration data batch → Run workflow, select the
   provider/model and explicitly approve the batch.
4. Confirm dry-run expense estimates are reasonable. Download the
   `seshat-calibration-<run>` ZIP from Artifacts after completion. The run may
   have fewer than 180 actual calls if its reserved spend cap is exhausted.
5. Inspect the validation report and count **verified, non-truncated real**
   observations. Failed and censored calls are not full regression labels.

## Spend and persistence limitations

This uses an ephemeral GitHub-hosted runner, so **each invocation is
independent**: the $2.00 cap applies to one run only and is not an account-wide
$10 monthly cap. Do not dispatch more than 5 batches without reconciling actual
spend and other Seshat/provider usage against the original $10 monthly target.
Avoid parallel unrelated API jobs; provider-side billing controls still matter.

Artifacts expire after 30 days. **They are NOT automatically merged into
durable training storage.** They must be securely consolidated into the canonical
storage root, de-duplicated, validated, and split by prompt family before
training. Do not train from exported CSV alone: normalized request/preflight/
observation metadata and raw provenance must remain linked.

The registered real-label trainer requires 100 verified uncensored training,
20 calibration and 20 test rows, across leakage-disjoint groups. Even 140 raw
successful responses may be inadequate due to censoring and split allocations.
This workflow does not auto-train or deploy an insufficient model.

## Adaptive batches (recommended from the second batch on)

The standard campaign above gives every category the same 512-token cap and
default effort. On `gpt-6.1-sol` that censored 36.7% of responses (66/180) and
produced no complete code-generation, code-review or verbose-explanation
labels, while ~half the spend went to truncated outputs. Adaptive batches fix
this with experimental design rather than volume:

1. Consolidate the previous artifacts (see [DATA_STORE.md](DATA_STORE.md)).
2. Plan (dry run, no requests):

   ```bash
   seshat-data plan-batch --store <store> --provider openai --model gpt-6.1-sol \
     --experiment-id adaptive_<yyyymmdd> \
     --output configs/experiments/batches/adaptive_<yyyymmdd>.yaml \
     --report configs/experiments/batches/adaptive_<yyyymmdd>.md
   ```

   Per category it fits a log-normal output-length model in which truncated
   responses are **right-censored** (lower bounds, never exact labels), picks
   the smallest cap in {512, 1024, 2048, 4096} with ≤10% predicted truncation
   at the highest planned effort, balances efforts `low/medium/high`, repeats
   10% of cells twice to measure run-to-run variance, and allocates requests by
   diminishing returns per `cost^0.5` with a per-category floor and share cap
   so long, realistic workloads stay represented. Expected spend targets 85% of
   the $2.00 cap. It refuses budgets above $2.00 and batches that would push the
   store's month-to-date estimated spend past the $10 target.
3. Open a PR with the YAML and the dry-run report. **Merging is the review;
   dispatching the workflow with `approve_paid_call=true` is the approval.**
4. Run *Manual calibration data batch* with `config_path` set to the YAML.
   `scripts/use_batch_config.py` refuses uncommitted configs, configs outside
   `configs/experiments/batches/`, provider/model mismatches, unverified models,
   caps above $2.00 and prompt/response text retention, and appends the run id
   to `experiment_id` so a re-dispatch never resumes another run's samples.
5. The consolidation workflow imports the artifact into the store and refreshes
   readiness, quality and spend reports.
