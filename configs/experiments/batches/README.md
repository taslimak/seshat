# Reviewed paid batch configs

Files here are produced by `seshat-data plan-batch` and run by the
*Manual calibration data batch* workflow via its `config_path` input.

* Commit each `<experiment>.yaml` together with its dry-run report
  (`<experiment>.md` / `.json`) in a pull request. Merging = reviewed.
* Paid execution additionally requires dispatching the workflow with
  `approve_paid_call=true`; the estimated spend is hard-capped at $2.00 per run.
* Never edit a config after it has run; plan a new experiment id instead.
