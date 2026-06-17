# Statistical Analysis

Confidence intervals were estimated with paired nonparametric bootstrap resampling over reports.
Macro F1 confidence intervals use the same bootstrap procedure as accuracy confidence intervals.
Paired model comparisons use exact McNemar tests for accuracy and bootstrap differences for macro F1.

## Metric Confidence Intervals

| Task | Method | n | Accuracy (95% CI) | Macro F1 (95% CI) | Errors |
| --- | --- | ---: | --- | --- | ---: |
| Cancer type | DeepSeek V4 Flash | 952 | 0.9800 (0.9695-0.9884) | 0.9778 (0.9660-0.9867) | 0 |
| Cancer type | DeepSeek V4 Pro | 952 | 0.9769 (0.9664-0.9863) | 0.9685 (0.9394-0.9827) | 0 |
| AJCC stage | DeepSeek V4 Flash | 594 | 0.8165 (0.7845-0.8468) | 0.7841 (0.7421-0.8201) | 0 |
| AJCC stage | DeepSeek V4 Pro | 594 | 0.8098 (0.7761-0.8401) | 0.6288 (0.5977-0.6573) | 14 |
| Prognosis | DeepSeek zero-shot | 100 | 0.5300 (0.4400-0.6300) | 0.5083 (0.4114-0.6033) | 0 |
| Prognosis | DeepSeek 8-shot | 100 | 0.5700 (0.4700-0.6700) | 0.5647 (0.4665-0.6599) | 0 |
| Prognosis | TF-IDF text-only logistic regression | 952 | 0.8571 (0.8351-0.8792) | 0.8543 (0.8312-0.8764) | 0 |
| Prognosis | TF-IDF + structured fields + DeepSeek variables | 952 | 0.8435 (0.8204-0.8666) | 0.8394 (0.8166-0.8630) | 0 |

## Paired Comparisons

| Comparison | n | Delta accuracy | McNemar b/c | McNemar p | Delta macro F1 (95% CI) | Bootstrap p |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Cancer type: Flash vs Pro | 952 | 0.0032 | 6/3 | 0.5078 | 0.0094 (-0.0048-0.0363) | 0.2330 |
| AJCC stage: Flash vs Pro | 594 | 0.0067 | 28/24 | 0.6778 | 0.1553 (0.1234-0.1843) | <0.0001 |
| Prognosis: TF-IDF text-only vs DeepSeek-assisted ML | 952 | 0.0137 | 54/41 | 0.2181 | 0.0149 (-0.0047-0.0349) | 0.1490 |

## Figure

![Macro F1 results](C:/Users/Mohamed KONE/Desktop/project/DeepSeekCancer/deepseek_pathrep/figures/macro_f1_results.png)
