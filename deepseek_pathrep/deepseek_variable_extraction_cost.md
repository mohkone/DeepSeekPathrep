# Cost Analysis

Pricing source: https://api-docs.deepseek.com/quick_start/pricing

If cache-hit token counts are absent, all prompt tokens are priced as cache misses.

| Run | Model | Reports | Correct | Tokens | Cost USD | Cost/report | Cost/1000 reports | Cost/correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek_variables_train_full.jsonl.metrics.json | deepseek-v4-flash | 7618 |  | 10968416 | $1.595526 | $0.000209 | $0.2094 |  |
| deepseek_variables_val_full.jsonl.metrics.json | deepseek-v4-flash | 953 |  | 1330250 | $0.090312 | $0.000095 | $0.0948 |  |
| deepseek_variables_test_full.jsonl.metrics.json | deepseek-v4-flash | 952 |  | 1386768 | $0.087333 | $0.000092 | $0.0917 |  |
