# Clinically Plausible Error Tables

These tables summarize recurring errors that are useful for the manuscript discussion.

## Rectum vs Colon Confusion

| Gold | Predicted | Count |
| --- | --- | ---: |
| Rectum adenocarcinoma | Colon adenocarcinoma | 5 |

| Index | ID | Cancer type | Raw stage | Gold | Predicted |
| ---: | --- | --- | --- | --- | --- |
| 139 | TCGA-AG-3902 | Rectum adenocarcinoma | Stage IIA | Rectum adenocarcinoma | Colon adenocarcinoma |
| 382 | TCGA-G5-6641 | Rectum adenocarcinoma | Stage IIIA | Rectum adenocarcinoma | Colon adenocarcinoma |
| 397 | TCGA-AF-6136 | Rectum adenocarcinoma | Stage IIIB | Rectum adenocarcinoma | Colon adenocarcinoma |
| 543 | TCGA-EI-7004 | Rectum adenocarcinoma |  | Rectum adenocarcinoma | Colon adenocarcinoma |
| 696 | TCGA-AG-3882 | Rectum adenocarcinoma | Stage I | Rectum adenocarcinoma | Colon adenocarcinoma |

## Kidney RCC Subtype Confusion

| Gold | Predicted | Count |
| --- | --- | ---: |
| Kidney renal clear cell carcinoma | Kidney chromophobe | 1 |
| Kidney renal papillary cell carcinoma | Kidney renal clear cell carcinoma | 3 |

| Index | ID | Cancer type | Raw stage | Gold | Predicted |
| ---: | --- | --- | --- | --- | --- |
| 406 | TCGA-BQ-5891 | Kidney renal papillary cell carcinoma | Stage III | Kidney renal papillary cell carcinoma | Kidney renal clear cell carcinoma |
| 436 | TCGA-BQ-5890 | Kidney renal papillary cell carcinoma | Stage III | Kidney renal papillary cell carcinoma | Kidney renal clear cell carcinoma |
| 550 | TCGA-AK-3433 | Kidney renal clear cell carcinoma | Stage II | Kidney renal clear cell carcinoma | Kidney chromophobe |
| 668 | TCGA-BQ-7045 | Kidney renal papillary cell carcinoma | Stage I | Kidney renal papillary cell carcinoma | Kidney renal clear cell carcinoma |

## Stage III vs Stage IV Confusion

| Gold | Predicted | Count |
| --- | --- | ---: |
| Stage III | Stage IV | 5 |
| Stage IV | Stage III | 24 |

| Index | ID | Cancer type | Raw stage | Gold | Predicted |
| ---: | --- | --- | --- | --- | --- |
| 19 | TCGA-GU-A42P | Bladder Urothelial Carcinoma | Stage IV | Stage IV | Stage III |
| 78 | TCGA-BR-7196 | Stomach adenocarcinoma | Stage IV | Stage IV | Stage III |
| 122 | TCGA-ZF-AA58 | Bladder Urothelial Carcinoma | Stage IV | Stage IV | Stage III |
| 181 | TCGA-DK-AA6Q | Bladder Urothelial Carcinoma | Stage IV | Stage IV | Stage III |
| 196 | TCGA-FD-A3B5 | Bladder Urothelial Carcinoma | Stage IV | Stage IV | Stage III |
| 245 | TCGA-BP-4787 | Kidney renal clear cell carcinoma | Stage IV | Stage IV | Stage III |
| 375 | TCGA-GU-A763 | Bladder Urothelial Carcinoma | Stage III | Stage III | Stage IV |
| 429 | TCGA-VQ-A91N | Stomach adenocarcinoma | Stage IV | Stage IV | Stage III |
| 459 | TCGA-QK-A8Z8 | Head and Neck squamous cell carcinoma | Stage IVC | Stage IV | Stage III |
| 463 | TCGA-BR-A453 | Stomach adenocarcinoma | Stage IV | Stage IV | Stage III |
| 483 | TCGA-BT-A20T | Bladder Urothelial Carcinoma | Stage IV | Stage IV | Stage III |
| 496 | TCGA-QK-A6IG | Head and Neck squamous cell carcinoma | Stage III | Stage III | Stage IV |
