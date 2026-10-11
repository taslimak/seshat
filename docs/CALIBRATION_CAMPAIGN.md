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
