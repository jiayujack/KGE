#!/bin/bash

# 定义所有要运行的模型
models=('autosf' 'boxe' 'compgcn' 'complex' 'complexliteral' 'conve' 'convkb' 'cooccurrencefiltered' 'cp' 'crosse')

# 设置其他不变的参数
data_dir="data/data_UNIPROT_or_GO_GO"
project_name="2_add_UNIPROT_GO"

# 循环遍历模型列表并依次运行训练命令
for model in "${models[@]}"
do
  echo "Running model: $model"
  
  # 运行训练命令并等待完成
  CUDA_VISIBLE_DEVICES=0 python train_classification.py \
    --model "$model" \
    --data_dir "$data_dir" \
    --project_name "$project_name" > output_$model.log 2>&1
  
  # 打印日志
  echo "Training for $model completed. Logs are saved in output_$model.log"
done

echo "All models have been trained sequentially."
