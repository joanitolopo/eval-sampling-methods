# eval-sampling-methods
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/joanitolopo/eval-sampling-methods/blob/main/LICENSE)

## Research Paper
If you find this code or research useful in your work, we kindly request that you cite the following paper: http://journal.uad.ac.id/index.php/JITEKI/article/view/25929/pdf_182.

```
@article{JITEKI25929,
	author = {Joanito Agili Lopo and Kristoko Dwi Hartomo},
	title = {Evaluating Sampling Techniques for Healthcare Insurance Fraud Detection in Imbalanced Dataset},
	journal = {Jurnal Ilmiah Teknik Elektro Komputer dan Informatika},
	volume = {9},
	number = {2},
	year = {2023},
	keywords = {Healthcare Insurance; Imbalanced Dataset; Oversampling; XGBoost; Fraud Detection; Undersampling},
	abstract = {Detecting fraud in the healthcare insurance dataset is challenging due to severe class imbalance, where fraud cases are rare compared to non-fraud cases. Various techniques have been applied to address this problem, such as oversampling and undersampling methods. However, there is a lack of comparison and evaluation of these sampling methods. Therefore, the research contribution of this study is to conduct a comprehensive evaluation of the different sampling methods in different class distributions, utilizing multiple evaluation metrics, including 𝐴𝑈𝐶𝑅𝑂𝐶, 𝐺 − 𝑚𝑒𝑎𝑛, 𝐹1𝑚𝑎𝑐𝑟o, Precision, and Recall. In addition, a model evaluation approach be proposed to address the issue of inconsistent scores in different metrics. This study employs a real-world dataset with the XGBoost algorithm utilized alongside widely used data sampling techniques such as Random Oversampling and Undersampling, SMOTE, and Instance Hardness Threshold. Results indicate that Random Oversampling and Undersampling perform well in the 50% distribution, while SMOTE and Instance Hardness Threshold methods are more effective in the 70% distribution. Instance Hardness Threshold performs best in the 90% distribution. The 70% distribution is more robust with the SMOTE and Instance Hardness Threshold, particularly in the consistent score in different metrics, although they have longer computation times. These models consistently performed well across all evaluation metrics, indicating their ability to generalize to new unseen data in both the minority and majority classes. The study also identifies key features such as costs, diagnosis codes, type of healthcare service, gender, and severity level of diseases, which are important for accurate healthcare insurance fraud detection. These findings could be valuable for healthcare providers to make informed decisions with lower risks. A well-performing fraud detection model ensures the accurate classification of fraud and non-fraud cases. The findings also can be used by healthcare insurance providers to develop more effective fraud detection and prevention strategies.},
	issn = {2338-3070},
	url = {http://journal.uad.ac.id/index.php/JITEKI/article/view/25929},
	pages = {223--238}
}
```
