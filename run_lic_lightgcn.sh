#!/bin/bash
# ==============================================================================
# A800 鍏ㄩ噺瀹為獙锛? 妯″瀷骞惰锛? 瀵规瘮 + 2 娑堣瀺锛?# 淇鍚庣殑 per-user 璇勪及鍗忚 + ReduceLROnPlateau
# ==============================================================================

set -e
export PYTHONUNBUFFERED=1

SEED=2026
LR=0.005
BATCH=4096
EPOCHS=100
DATA_DIR="./"
TB_LOG_DIR="/root/tf-logs"
COMMON_ARGS="--seed=${SEED} --lr=${LR} --bpr_batch_size=${BATCH} --epochs=${EPOCHS} --patience=15 --data_dir=${DATA_DIR} --use_amp --tb_log_dir=${TB_LOG_DIR} --eval_every=1 --max_test_samples=0 --min_user_inter=100 --min_item_inter=800"

echo "=========================================="
echo "  A800 鍏ㄩ噺瀹為獙 (5 妯″瀷骞惰)"
echo "  Seed=${SEED}  LR=${LR}  Batch=${BATCH}"
echo "=========================================="

mkdir -p logs

# --- 3 涓姣旀ā鍨?---
echo "[鍚姩] 鍌呴噷鍙舵ā鍨?(fusion=concat)"
python main.py --temporal_model=fourier --fusion_mode=concat ${COMMON_ARGS} \
    > logs/log_fourier_concat.txt 2>&1 &
PID_FOURIER_CONCAT=$!

echo "[鍚姩] 楂樻柉鍏磋叮鏃堕挓"
python main.py --temporal_model=gaussian ${COMMON_ARGS} \
    > logs/log_gaussian.txt 2>&1 &
PID_GAUSSIAN=$!

echo "[鍚姩] LIC 闀挎湡鍏磋叮鏃堕挓"
python main.py --temporal_model=lic ${COMMON_ARGS} \
    > logs/log_lic.txt 2>&1 &
PID_LIC=$!

# --- 2 涓倕閲屽彾铻嶅悎娑堣瀺 ---
echo "[鍚姩] 鍌呴噷鍙舵ā鍨?(fusion=add)"
python main.py --temporal_model=fourier --fusion_mode=add ${COMMON_ARGS} \
    > logs/log_fourier_add.txt 2>&1 &
PID_ADD=$!

echo "[鍚姩] 鍌呴噷鍙舵ā鍨?(fusion=mlp)"
python main.py --temporal_model=fourier --fusion_mode=mlp ${COMMON_ARGS} \
    > logs/log_fourier_mlp.txt 2>&1 &
PID_MLP=$!

echo ""
echo "5 涓ā鍨嬪凡鍚庡彴鍚姩:"
echo "  鍌呴噷鍙?concat)  PID=${PID_FOURIER_CONCAT}"
echo "  楂樻柉鍏磋叮鏃堕挓    PID=${PID_GAUSSIAN}"
echo "  LIC             PID=${PID_LIC}"
echo "  鍌呴噷鍙?add)     PID=${PID_ADD}"
echo "  鍌呴噷鍙?mlp)     PID=${PID_MLP}"
echo ""
echo "鐩戞帶: tail -f logs/log_*.txt"

wait ${PID_FOURIER_CONCAT} ${PID_GAUSSIAN} ${PID_LIC} ${PID_ADD} ${PID_MLP}

echo "=========================================="
echo "  鍏ㄩ儴 5 涓疄楠屽凡瀹屾垚"
echo "=========================================="
