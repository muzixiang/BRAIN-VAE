#!/bin/bash

# ====================== 定义文件路径 ======================

# 输入数据目录
DATA_DIR="./data"


# 基础输出目录
BASE_OUTPUT_DIR="output/brain-vae_PER"
# 子目录
RESULTS_DIR="${BASE_OUTPUT_DIR}/results"
PLOTS_DIR="${BASE_OUTPUT_DIR}/plots"
LOGS_DIR="${BASE_OUTPUT_DIR}/logs"

# ====================== 确保目录存在 ======================
mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$LOGS_DIR"

# 获取当前时间并格式化
current_time=$(date +"%Y%m%d_%H%M%S")

# 定义文件列表
files=(
    "TUR4_1" "TUR4_2" "TUR4_3" "TUR4_4" "TUR4_5"
    "TEMP_1" "TEMP_2" "TEMP_3" "TEMP_4" "TEMP_5"
    "PSAL_1" "PSAL_2" "PSAL_3" "PSAL_4" "PSAL_5"
    "WSPD_1" "WSPD_2" "WSPD_3" "WSPD_4" "WSPD_5"
    "CPHL_1" "CPHL_2" "CPHL_3" "CPHL_4" "CPHL_5"
    "DOX1_1" "DOX1_2" "DOX1_3" "DOX1_4" "DOX1_5"
)

# files=(
#     "TUR4_1" "TUR4_2" "TUR4_3" "TUR4_4" "TUR4_5"
#     "TEMP_1" "TEMP_2" "TEMP_3" "TEMP_4" "TEMP_5"
#     "PSAL_1" "PSAL_2" "PSAL_3" "PSAL_4" "PSAL_5"
# )

# 测试用简化文件列表（取消下面注释可快速测试）
# files=("TUR4_1")


# 定义参数

# 先运行的预测长度
PHASE1_PRED_LENGTHS=(24 48)
# 后运行的预测长度（单步）
PHASE2_PRED_LENGTHS=(1)

# prediction_lengths=(1 24 48)  # 预测长度列表 (根据你的实验设置调整)
hw=60                  # 历史窗口大小
h_dim=128              # 隐藏层维度
Z_dim=8                # 潜在空间维度
lr=0.001               # 学习率 (与python脚本默认值一致)
priority_lr=0.001      # 优先级网络学习率
train_ratio=0.3        # 预热数据比例
mb_size=1              # 在线学习的batch size (与python脚本默认值一致)
# use_amp=true           # 是否使用混合精度训练
device="cuda:0"        # 使用设备
encoder_depth=1        # TSEncoder模块深度

# === REPLAY 参数 (新增加) ===
replay_frequency=1                  # 每处理多少批次进行一次重放
replay_times=1                      # 每次经验重放的迭代次数
priority_updates=2                  # 每次重放时更新PriorityNetwork的次数
top_k=10                            # 每次选择Top-K的经验重放
max_storage_size=100000             # 重放缓冲区最大容量
priority_input_dim=$((4*8+3))       # 优先级网络输入维度（4*8+3）
output_prefix="replay_PER"           # 输出文件前缀

# echo "Starting BRAIN-VAE-PER model online learning and inference..."
# echo "===================================================="
# echo "Base output directory: ${BASE_OUTPUT_DIR}"
# echo "Current time: ${current_time}"
# echo "Total files to process: ${#files[@]}"
# echo "Prediction lengths: ${prediction_lengths[@]}"
# echo "===================================================="

# # 遍历文件和预测长度
# for file in "${files[@]}"; do
#     for pw in "${prediction_lengths[@]}"; do
#         echo "----------------------------------------------------"
#         echo "Processing file: $file"
#         echo "Historical Length: $hw | Prediction Length: $pw"
#         echo "Model Parameters: h_dim=$h_dim, Z_dim=$Z_dim, lr=$lr"
        
#         # # 构建AMP参数
#         # amp_flag=""
#         # if [ "$use_amp" = true ]; then
#         #     amp_flag="--use_amp"
#         # fi

#         # 运行 Python 脚本
#         python -u brain-vae-PER.py \
#             --seed 42 \
#             --data_dir "$DATA_DIR" \
#             --filename "$file" \
#             --train_ratio "$train_ratio" \
#             --hw "$hw" \
#             --pw "$pw" \
#             --h_dim "$h_dim" \
#             --Z_dim "$Z_dim" \
#             --priority_input_dim "$priority_input_dim" \
#             --encoder_depth "$encoder_depth" \
#             --mb_size "$mb_size" \
#             --lr "$lr" \
#             --priority_lr "$priority_lr" \
#             --replay_frequency "$replay_frequency" \
#             --replay_times "$replay_times" \
#             --priority_updates "$priority_updates" \
#             --top_k "$top_k" \
#             --max_storage_size "$max_storage_size" \
#             --device "$device" \
#             --results_dir "$RESULTS_DIR" \
#             --plots_dir "$PLOTS_DIR" \
#             --output_prefix "$output_prefix" \
#             > "${LOGS_DIR}/logs_${file}_hw${hw}_pw${pw}_${current_time}.log" 2>&1
        
#         # 检查命令执行状态
#         if [ $? -eq 0 ]; then
#             echo "Successfully processed ${file} with pw=${pw}"
#         else
#             echo "Error occurred while processing ${file} with pw=${pw}"
#             echo "Check log file: ${LOGS_DIR}/logs_${file}_hw${hw}_pw${pw}_${current_time}.log"
#         fi
        
#         # 添加间隔以避免资源争用
#         # sleep 1
#     done
# done

# echo "===================================================="
# echo "All tasks completed!"
# echo "Results saved to: ${RESULTS_DIR}"
# echo "Plots saved to: ${PLOTS_DIR}"
# echo "Logs saved to: ${LOGS_DIR}"
# echo "===================================================="



# ========================== 先 24、48 后 1 ===========================

echo "Starting BRAIN-VAE-PER model online learning and inference..."
echo "Base output directory: ${BASE_OUTPUT_DIR}"
echo "Current time: ${current_time}"

# ==============================================
# ============== 阶段一：24 和 48 步 =============
# ==============================================
echo "===================================================="
echo "Phase 1: Running prediction lengths: ${PHASE1_PRED_LENGTHS[@]}"
echo "----------------------------------------------------"

for file in "${files[@]}"; do
    for pw in "${PHASE1_PRED_LENGTHS[@]}"; do
        echo "Processing file: $file | Prediction Length: $pw"
        python -u brain-vae-PER.py \
            --seed 42 \
            --data_dir "$DATA_DIR" \
            --filename "$file" \
            --train_ratio "$train_ratio" \
            --hw "$hw" \
            --pw "$pw" \
            --h_dim "$h_dim" \
            --Z_dim "$Z_dim" \
            --priority_input_dim "$priority_input_dim" \
            --encoder_depth "$encoder_depth" \
            --mb_size "$mb_size" \
            --lr "$lr" \
            --priority_lr "$priority_lr" \
            --replay_frequency "$replay_frequency" \
            --replay_times "$replay_times" \
            --priority_updates "$priority_updates" \
            --top_k "$top_k" \
            --max_storage_size "$max_storage_size" \
            --device "$device" \
            --results_dir "$RESULTS_DIR" \
            --plots_dir "$PLOTS_DIR" \
            --output_prefix "$output_prefix" \
            > "${LOGS_DIR}/logs_${file}_hw${hw}_pw${pw}_${current_time}.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "Successfully processed ${file} with pw=${pw}"
        else
            echo "Error occurred while processing ${file} with pw=${pw}"
        fi
    done
done

# 等待所有 Phase 1 任务完成后再继续
echo "Phase 1 completed."

# ==============================================
# ============== 阶段二：1 步预测 ================
# ==============================================
echo "===================================================="
echo "Phase 2: Running prediction length: ${PHASE2_PRED_LENGTHS[0]}"
echo "----------------------------------------------------"

for file in "${files[@]}"; do
    for pw in "${PHASE2_PRED_LENGTHS[@]}"; do
        echo "Processing file: $file | Prediction Length: $pw"
        python -u brain-vae-PER.py \
            --seed 42 \
            --data_dir "$DATA_DIR" \
            --filename "$file" \
            --train_ratio "$train_ratio" \
            --hw "$hw" \
            --pw "$pw" \
            --h_dim "$h_dim" \
            --Z_dim "$Z_dim" \
            --priority_input_dim "$priority_input_dim" \
            --encoder_depth "$encoder_depth" \
            --mb_size "$mb_size" \
            --lr "$lr" \
            --priority_lr "$priority_lr" \
            --replay_frequency "$replay_frequency" \
            --replay_times "$replay_times" \
            --priority_updates "$priority_updates" \
            --top_k "$top_k" \
            --max_storage_size "$max_storage_size" \
            --device "$device" \
            --results_dir "$RESULTS_DIR" \
            --plots_dir "$PLOTS_DIR" \
            --output_prefix "$output_prefix" \
            > "${LOGS_DIR}/logs_${file}_hw${hw}_pw${pw}_${current_time}.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "Successfully processed ${file} with pw=${pw}"
        else
            echo "Error occurred while processing ${file} with pw=${pw}"
        fi
    done
done

echo "===================================================="
echo "All tasks completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "Plots saved to: ${PLOTS_DIR}"
echo "Logs saved to: ${LOGS_DIR}"
echo "===================================================="