# Cost Analysis

Pricing source: https://api-docs.deepseek.com/quick_start/pricing

If cache-hit token counts are absent, all prompt tokens are priced as cache misses.

| Run | Model | Reports | Correct | Tokens | Cost USD | Cost/report | Cost/1000 reports | Cost/correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek_cancer_type_full.jsonl.metrics.json | deepseek-v4-flash | 952 | 933 | 1292828 | $0.182611 | $0.000192 | $0.1918 | $0.000196 |
| deepseek_v4_pro_cancer_type_full.jsonl.metrics.json | deepseek-v4-pro | 952 | 930 | 1294297 | $0.463766 | $0.000487 | $0.4871 | $0.000499 |
| deepseek_ajcc_stage_full_v2_max4096.jsonl.metrics.json | deepseek-v4-flash | 594 | 485 | 945862 | $0.161103 | $0.000271 | $0.2712 | $0.000332 |
| deepseek_v4_pro_ajcc_stage_full_max4096.jsonl.metrics.json | deepseek-v4-pro | 594 | 481 | 1170725 | $0.695980 | $0.001172 | $1.1717 | $0.001447 |
| deepseek_prognosis_100_max4096.jsonl.metrics.json | deepseek-v4-flash | 100 | 53 | 151124 | $0.024516 | $0.000245 | $0.2452 | $0.000463 |
| deepseek_prognosis_100_fewshot8_max4096.jsonl.metrics.json | deepseek-v4-flash | 100 | 57 | 386105 | $0.058671 | $0.000587 | $0.5867 | $0.001029 |
