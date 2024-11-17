import deg
import pick_deg
import train_gtex_all_pls as train
import sys
import test_gtex_train
import numpy as np
import common_test

split_id = sys.argv[1]

for threshold in np.arange(30, 300, 10):
    deg.main(threshold)
    pick_deg.main(threshold)
    sys.argv = ['common_test.py', 'deg', split_id]
    common_test.main()
    sys.argv = ['train_gtex_all_pls.py', 'deg', '20', f"cmn{split_id}"]
    train.main()
    print("Testing on threshold =", threshold)
    sys.argv = ['test_gtex_train.py', 'deg', '20', f"cmn{split_id}", "pls"]
    split_res = test_gtex_train.main(
        main=True, 
        md_hot_organ=None,
        tissue=None,
        delete_model=False,
        deg_th=str(threshold)
    )