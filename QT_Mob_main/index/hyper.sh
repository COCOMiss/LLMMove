#!/bin/bash
# ============================================================
# run_grid_search.sh
# 多超参数组合批量实验：beta × quant_loss_weight × lr × dropout × num_emb_list × sk_epsilon
# ============================================================

set -e
set -o pipefail

# ====== 环境配置 ======
PROJECT_DIR="/home/linyuxi/LLM/LLMMove/QT_Mob_main/index"
PYTHON_BIN="python3"


# ====== 输出目录 ======
LOG_DIR="${PROJECT_DIR}/logs_grid"
mkdir -p "$LOG_DIR"

# ====== 超参数列表 ======
betas=(0.1 0.25 0.5)
quant_loss_weights=(1.0 1.5)
lrs=(5e-4 1e-4 1e-3)

# ====== 实验循环 ======
for beta in "${betas[@]}"; do
  for qlw in "${quant_loss_weights[@]}"; do
    for lr in "${lrs[@]}"; do
            EXP_NAME="b${beta}_q${qlw}_lr${lr}_d${dropout}_emb$(echo $emb_list | tr -d '[], ')_eps$(echo $eps | tr -d '[], ')"
            CKPT_DIR="liandanlu/${EXP_NAME}"
            LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

            echo "======================================================"
            echo "🚀 Running experiment:"
            echo "  beta=${beta}"
            echo "  quant_loss_weight=${qlw}"
            echo "  lr=${lr}"
            echo "  ckpt_dir=${CKPT_DIR}"
            echo "======================================================"

            $PYTHON_BIN "${PROJECT_DIR}/main.py" \
              --beta "${beta}" \
              --quant_loss_weight "${qlw}" \
              --lr "${lr}" \
              --ckpt_dir "${CKPT_DIR}" \
              2>&1 | tee "${LOG_FILE}"

            echo "✅ Finished: ${EXP_NAME}"
            echo ""
          done
        done
      done
    done
  done
done

echo "🎯 All grid experiments completed! Logs are in ${LOG_DIR}"
