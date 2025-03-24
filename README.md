# Modeling Tissue-Specific Aging Using Machine Learning

This repository contains the codebase for the undergraduate thesis titled *Modeling Tissue-Specific Aging Using Machine Learning*, conducted by [**Wasif Jalal**](https://github.com/wjalal) and [**Mubasshira Musarrat**](https://github.com/MubasshiraMusarrat) at the **Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology**, under the supervision of **Dr. M. Sohel Rahman**.

## Data Requirements
Place the necessary tissue-specific expression data from [Adult GTEx v10](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression#bulk_tissue_expression-gtex_analysis_v10-rna-seq-Gene_TPMs_by_tissue-container) in the following directories:

- **TPM Expression Data**: `gtex/datav10`
- **Read Count Data**: `gtex/data`

To process the data into our study's format, run `proc/gtexv10_to_organage.sh`.

## Workflow

### 1. Initial Clustering
Run the following scripts:
- `gcttoCsv.py`
- Scripts in `clustering/all_organ` and `clustering/per_organ`

### 2. Optimal Fixed-Point Interpolation
- `true_age_interpolation.py`
- `true_age_output_view.py`

### 3. Feature Selection
#### (1) Correlation-Based Selection
- `pick_genes.py`

#### (2) DeSeq-Based Selection
- `deg_thresholding.py`
- `deg_thresh_finetune.py`
- Set the optimal threshold from finetuning results in `pick_deg_optim.py`

#### (3) Oh et al. Method
- `identify_organ_enriched_genes.py`
  - Note: **Artery Coronary** and **Aorta** are treated as one organ.

### 4. Train-Test Splitting
- `stratified_split_dthhrdy.py`

### 5. Model Training and Testing
- `train_gtex_all_<regr>.py`
- `test_gtex_train.py`
- `tissue_agegap_analytics_multi.py`
- Run: `stf_sp_train_test_multi.sh`

### 6. Leave-P-Out Train-Test for Downstream Analyses
- Run: `lpo_coeff_multi.sh`
- `all_agegap_analytics_multi.py`
- `agegap_lpo_stats.py`

## Usage
To reproduce the experiments, follow the steps outlined above in the correct order. Adjust script parameters as needed for specific analyses.

---
For any questions, feel free to open an issue or reach out!
