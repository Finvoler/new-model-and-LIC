#!/bin/bash
# 蹇€熼獙璇佽剼鏈細涓ユ牸绛涢€?鈫?~7.6K 鐗╁搧, ~6.7K 鐢ㄦ埛
# item>=1000 鍗曡疆绛涚墿鍝? user>=100 鍗曡疆绛涚敤鎴? lr=0.005
# 姣?epoch 閮借瘎浼?+ AUC 鎸囨爣

set -e
cd /root/new

COMMON="--seed=2026 --lr=0.005 --bpr_batch_size=4096 --epochs=10 \
--data_dir=/root/new/ --use_amp \
--eval_every=1 --max_test_samples=20000 --test_u_batch_size=500 \
--min_user_inter=100 --min_item_inter=1000 \
--tb_log_dir=/root/tf-logs/validate"

echo "============================================"
echo "  蹇€熼獙璇侊細Fourier-concat (杩蜂綘鏁版嵁闆?"
echo "============================================"
python main.py --temporal_model=fourier --fusion_mode=concat $COMMON 2>&1

echo ""
echo "============================================"
echo "  蹇€熼獙璇侊細Gaussian (杩蜂綘鏁版嵁闆?"
echo "============================================"
python main.py --temporal_model=gaussian $COMMON 2>&1

echo ""
echo "============================================"
echo "  蹇€熼獙璇侊細LIC (杩蜂綘鏁版嵁闆?"
echo "============================================"
python main.py --temporal_model=lic --lic_max_behaviors=100 --lic_top_k=30 $COMMON 2>&1

echo ""
echo "============================================"
echo "  鍏ㄩ儴楠岃瘉瀹屾垚锛?
echo "============================================"
