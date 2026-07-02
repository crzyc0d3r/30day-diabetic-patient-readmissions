# Champion F1 impact under executed drift scenarios

Reference: `data/features.csv` (99,340 rows). Scorer: @champion (`final_model.joblib` + `full_inference_pipeline.joblib`). Each scenario batch is genuinely labeled, so the F1 delta is **measured harm**, not an estimate. Verdict bands (PSI WARN/ALERT) come from `helpers.constants`.

| scenario | verdict | current_rows | ref_f1 | cur_f1 | delta_f1 | ref_precision | cur_precision | delta_precision | ref_recall | cur_recall | delta_recall | ref_auc_roc | cur_auc_roc | delta_auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casemix_shift | ALERT | 99340 | 0.3271 | 0.3206 | -0.0065 | 0.2300 | 0.2241 | -0.0059 | 0.5662 | 0.5628 | -0.0034 | 0.7299 | 0.7223 | -0.0076 |
| coding_shift | ALERT | 99340 | 0.3271 | 0.3280 | +0.0009 | 0.2300 | 0.2313 | +0.0013 | 0.5662 | 0.5636 | -0.0026 | 0.7299 | 0.7256 | -0.0043 |
| current | OK | 99340 | 0.3271 | 0.3265 | -0.0007 | 0.2300 | 0.2297 | -0.0003 | 0.5662 | 0.5638 | -0.0024 | 0.7299 | 0.7304 | +0.0005 |
| formulary_shift | ALERT | 99340 | 0.3271 | 0.3235 | -0.0036 | 0.2300 | 0.2315 | +0.0015 | 0.5662 | 0.5369 | -0.0293 | 0.7299 | 0.7235 | -0.0064 |
| los_utilization_shift | ALERT | 99340 | 0.3271 | 0.3109 | -0.0163 | 0.2300 | 0.2239 | -0.0061 | 0.5662 | 0.5083 | -0.0579 | 0.7299 | 0.7121 | -0.0178 |
| mixed_severe | ALERT | 99340 | 0.3271 | 0.3080 | -0.0191 | 0.2300 | 0.2215 | -0.0085 | 0.5662 | 0.5053 | -0.0609 | 0.7299 | 0.7012 | -0.0287 |
| none | OK | 99340 | 0.3271 | 0.3265 | -0.0007 | 0.2300 | 0.2297 | -0.0003 | 0.5662 | 0.5638 | -0.0024 | 0.7299 | 0.7304 | +0.0005 |
