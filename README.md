Place tissue-specific TPM expression data from [Adult GTEx v10](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression#bulk_tissue_expression-gtex_analysis_v10-rna-seq-Gene_TPMs_by_tissue-container) in `gtex/datav10`

Place tissue-specific read count data from [Adult GTEx v10](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression#bulk_tissue_expression-gtex_analysis_v10-rna-seq-Gene_read_counts_by_tissue-container) in `gtex/data`

Initial Clustering
    Run gcttoCsv.py
    Run the scripts in clustering/all_organ and clustering/per_organ

Optimal Fixed point interpolation
    true_age_interpolation.py
    true_age_output_view.py

Feature Selection
    1) Correlation: pick_genes.py
    2) DeSeq:
        deg_thresholding.py
        deg_thresh_finetune.py
        set optimal threshold from results of finetuning, in pick_deg_optim.py
    3) Oh et. al.
        identify_organ_enriched_genes.py
        ////artery_coronary , aorta together as one organ

Train-Test Splitting
    stratified_split_dthhrdy.py

Model Training and Testing 
    train_gtex_all_<regr>.py
    test_gtex_train.py
    tissue_agegap_analytics_multi.py
    > stf_sp_train_test_multi.sh

Leave-P-Out Train-Test for Downstream Analyses
    > lpo_coeff_multi.sh
    all_agegap_analytics.py
    agegap_lpo_stats.py