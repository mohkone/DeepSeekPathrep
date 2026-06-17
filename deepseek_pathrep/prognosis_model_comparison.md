# Prognosis Model Comparison

Baseline macro F1: `0.8543`

| Run | Model | Accuracy | Macro F1 | Delta vs baseline | Train rows | Test rows | DeepSeek train features | DeepSeek test features | Warning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ml_prognosis_tfidf_logreg_test.jsonl.metrics.json | tfidf_logistic_regression | 0.8529 | 0.8501 | -0.0042 | 8566 | 952 |  |  |  |
| ml_prognosis_text_only_test.jsonl.metrics.json | tfidf_logistic_regression | 0.8571 | 0.8543 | +0.0000 | 8566 | 952 |  |  |  |
| ml_prognosis_deepseek_assisted_test.jsonl.metrics.json | tfidf_logistic_regression | 0.8435 | 0.8394 | -0.0149 | 8566 | 952 | 8558 | 951 |  |
| ml_prognosis_deepseek_structured_only_test.jsonl.metrics.json | tfidf_logistic_regression | 0.6607 | 0.6512 | -0.2031 | 8566 | 952 | 8558 | 951 |  |
| ml_prognosis_deepseek_only_test.jsonl.metrics.json | tfidf_logistic_regression | 0.6492 | 0.6407 | -0.2136 | 8566 | 952 | 8558 | 951 |  |
| ml_prognosis_base_structured_only_test.jsonl.metrics.json | tfidf_logistic_regression | 0.5578 | 0.5540 | -0.3003 | 8566 | 952 |  |  |  |
