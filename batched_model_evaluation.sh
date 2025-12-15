# 设置 GPU ID，例如：7
GPU_ID=0

# base_sparsity 范围：从 20 到 44（含）
START_SPARSITY=32
END_SPARSITY=32

# 固定参数
SPARSE_MODEL_PATH=""


# for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
for BASE_SPARSITY in $(seq $END_SPARSITY -1 $START_SPARSITY); do
    echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
    MODEL_PATH="${SPARSE_MODEL_PATH}/${BASE_SPARSITY}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval \
        --model hf \
        --model_args pretrained="${MODEL_PATH}"\
        --task boolq,ai2_arc,openbookqa,winogrande,hellaswag,rte\
    
    echo "Done base_sparsity=${BASE_SPARSITY}"
    echo "---------------------------------------------"
done

# for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
for BASE_SPARSITY in $(seq $END_SPARSITY -1 $START_SPARSITY); do
    echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
    MODEL_PATH="${SPARSE_MODEL_PATH}/${BASE_SPARSITY}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval \
        --model hf \
        --model_args pretrained="${MODEL_PATH}"\
        --task mmlu \
    
    echo "Done base_sparsity=${BASE_SPARSITY}"
    echo "---------------------------------------------"
done

# for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
for BASE_SPARSITY in $(seq $END_SPARSITY -1 $START_SPARSITY); do
    echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
    MODEL_PATH="${SPARSE_MODEL_PATH}/${BASE_SPARSITY}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval \
        --model hf \
        --model_args pretrained="${MODEL_PATH}"\
        --task gsm8k \
    
    echo "Done base_sparsity=${BASE_SPARSITY}"
    echo "---------------------------------------------"
done
