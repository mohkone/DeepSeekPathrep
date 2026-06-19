Dear Editors,

We are pleased to submit our manuscript, "DeepSeek for Pathology Report Understanding: A Benchmark Study of Cancer Type Extraction, AJCC Staging, and Prognosis Prediction," for consideration in BMC Bioinformatics.

This study provides an independent benchmark of DeepSeek on PathRep-Bench pathology report understanding tasks derived from TCGA pathology reports. We evaluate cancer type extraction, AJCC staging, and prognosis prediction, and we compare DeepSeek V4 Flash and DeepSeek V4 Pro under a reproducible prompting and evaluation framework.

The manuscript makes four main contributions. First, it shows that DeepSeek achieves near-ceiling performance for cancer type extraction and strong performance for AJCC staging from pathology reports. Second, it provides a practical cost-performance comparison of DeepSeek V4 Flash and DeepSeek V4 Pro, showing that the cheaper Flash model offered the better cost-performance tradeoff in the evaluated setting. Third, it reports confidence intervals and paired statistical comparisons, including McNemar tests and bootstrap macro F1 comparisons, to support the interpretation of model differences. Fourth, it demonstrates that prognosis prediction remains better framed as supervised outcome modeling: DeepSeek prompting was weak for prognosis, and DeepSeek-extracted structured variables did not significantly improve a TF-IDF supervised prognosis model.

Together, these findings support a hybrid architecture for pathology report understanding: DeepSeek can be used for flexible information extraction and staging support, while prognosis prediction should be handled by dedicated supervised models trained directly against outcome labels. We believe this practical distinction will be useful to researchers developing clinical NLP pipelines and to readers interested in the appropriate role of LLMs in biomedical information extraction and prediction tasks.

The source code and reproducible analysis scripts are available at https://github.com/mohkone/DeepSeekPathrep. The archived version corresponding to this manuscript is available through Zenodo at https://doi.org/10.5281/zenodo.20754374.

This manuscript has not been published and is not under consideration elsewhere. The authors declare no competing interests.

Sincerely,

Mohamed Kone  
School of Computer Science and Engineering, Hunan University  
Changsha, China  
Email: mohamed.kone.374@gmail.com

On behalf of all authors:
Mohamed Kone, Shulin Wang, and Gaoussou Haidara
