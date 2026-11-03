# Tracing-the-Logical-Progression-in-Introductions-and-Its-Scholarly-Impact

Data and code for the paper:
 
> Yang, Y., Peng, H., & Zhang, C. Tracing the logical progression in introductions and its scholarly impact. *Journal of Informetrics* (under revision). Preprint: https://ssrn.com/abstract=6672666
 
An earlier version of this work received the Best Student Short Paper Award at ICGAL 2025.
 
## Overview
 
We use a large language model (LLM), prompted with established writing guidelines, to label the logical function of each paragraph in the Introduction of 125,825 research articles in six disciplines (Biology, Computer Science, Economics, Mathematics, Medicine and Physics). We then relate the resulting logical-progression measures to citation outcomes, controlling for paper, author, journal and topic characteristics.
 
This repository contains the scripts for each step and the processed data needed to reproduce the analyses. Full texts are not redistributed; they can be retrieved from S2ORC using the paper IDs provided.
 
## Repository structure
 
```
├── Code/                        Python scripts (see "Code" below)
├── ChatGPT_writing_score/       LLM paragraph labels and derived measures, by field
├── MAG_journal_info/            Journal identifiers and journal citation counts, by field
├── MAG_paper_author_topic/      Paper, author and topic variables used as controls, by field
├── S2ORC_field_paper_info/      Basic metadata of the sampled papers, by field
└── s2orc_introduction_journal_paper_id.txt   IDs of all papers with an identifiable Introduction
```
 
All data files are semicolon-separated text files with a header row. Papers are linked across files by `s2orc_paper_id` (S2ORC) and `mag_id` (Microsoft Academic Graph).
 
## Data
 
### `ChatGPT_writing_score/`
 
**`{Field}_writing_score.txt`** – raw LLM labels, one row per paper.
The `writing_score` column lists one `goal:support` pair per Introduction paragraph, in order. For example, `1:3,2:2,4:3` describes a three-paragraph Introduction.
 
| Code | Paragraph goal | | Code | Degree of support |
|---|---|---|---|---|
| 0 | Not sure | | 1 | Weak |
| 1 | Introduce the topic (IT) | | 2 | Moderate |
| 2 | Describe the background (DB) | | 3 | Strong |
| 3 | Establish the research problem (ERQ) | | | |
| 4 | Specify the objectives (SO) | | | |
| 5 | Summarize the findings (SF) | | | |
| 6 | Map out the paper (MO) | | | |
 
**`{Field}_writing_score_normalization.txt`** – measures derived from the raw labels: the number of paragraphs (`len_paragraph`), the share of paragraphs serving each goal (`IT`, `DB`, `ERQ`, `SO`, `SF`, `MO`), and the distribution of each goal across six relative positions in the Introduction (e.g. `DB_0_17` = background in the first 17% of paragraphs).
 
### `MAG_journal_info/`
`{Field}_journal_cc.txt`: `s2orc_paper_id`, `mag_id`, `journal_id`, `journal_cc` (journal citation count).
 
### `MAG_paper_author_topic/`
`{Field}_paper_author_topic.txt`: publication year, number of authors, citation and reference counts, topic size, productivity and h-index of the first and last authors, and title length.
 
### `S2ORC_field_paper_info/`
`{Field}_journal_paper_reference_6paras.txt`: year, MAG ID, number of authors, citation and reference counts, and title of the sampled papers.
 
## Code
 
Run the scripts in the following order. Paths at the top of each script point to local copies of S2ORC and MAG and need to be changed.
 
| Step | Script | Purpose |
|---|---|---|
| 1 | `Data_collection.py` | Extract journal articles with an identifiable Introduction from S2ORC and link them to MAG |
| 2 | `MAGDataEnricher.py` | Add paper, author, journal and topic variables from MAG |
| 3 | `LLM_writing.py` | Label Introduction paragraphs with the LLM (main analysis, `gpt-3.5-turbo`) |
| 4 | `LLM_writing_v2.py` | Re-label a test set with other LLMs for robustness checks (`gpt-5.2`, `gemini-3-flash-preview`) |
| 5 | `Evaluation_ML_GPT.py`, `Evaluation_DL.py`, `Evaluation_LLMs.py` | Compare LLM labels with human annotations, machine-learning and BERT baselines, and across LLMs |
| 6 | `Adherence.py` | Measure coverage of goals, adherence to the guideline sequence and the share of unclear paragraphs |
| 7 | `Regression.py` | Descriptive statistics and negative binomial regressions of citations |
| 8 | `Group_comparison.py` | Propensity score matching and group comparisons |
| 9 | `Visualization.py` | Figures |
 
### Requirements
 
Python 3.9+ with `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `pingouin`, `matplotlib`, `seaborn`, `torch`, `transformers`, `openai`, `httpx` and `tqdm`.
 
```bash
pip install pandas numpy scipy statsmodels scikit-learn pingouin matplotlib seaborn torch transformers openai httpx tqdm
```
 
The labelling scripts read the API key from an environment variable:
 
```bash
export OPENAI_API_KEY="your-key"
```
 
## Data sources and licences
 
- **S2ORC** (Lo et al., 2020): https://github.com/allenai/s2orc, released under ODC-By 1.0.
- **Microsoft Academic Graph** (Sinha et al., 2015), released under ODC-By 1.0.
Derived data in this repository are shared under the same terms. Code is released under the MIT License (see `LICENSE`).
 
## Citation
 
If you use this code or data, please cite:
 
```bibtex
@article{yang_logical_progression,
  author  = {Yang, Yunhan and Peng, Hao and Zhang, Chenwei},
  title   = {Tracing the Logical Progression in Introductions and Its Scholarly Impact},
  journal = {Journal of Informetrics},
  note    = {Under revision. Preprint: https://ssrn.com/abstract=6672666}
}
```
 
## Contact
 
Yunhan Yang, University of Sheffield – yunhan.yang@sheffield.ac.uk
