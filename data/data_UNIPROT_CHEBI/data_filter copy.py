import pandas as pd
import os
from sklearn.model_selection import train_test_split
#import ipdb;ipdb.set_trace()

LHS_TYPE="UNIPROT"
RHS_TYPE="CHEBI"
REL="catalytic_activity_CHEBI"
SAVE_DIR="./data_"+LHS_TYPE+"_"+RHS_TYPE
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 读取 parquet 文件
data = pd.read_parquet('./PSI_kg_full.parquet')

# 过滤出符合条件的行：lhs_type 为 UNIPROT，rhs_type 为 CHEBI，rel 为 catalytic_activity_CHEBI
filtered_data = data[
    (data['lhs_type'] == LHS_TYPE) &
    (data['rhs_type'] == RHS_TYPE) &
    (data['rel'] == REL)
]

# 拆分数据集为 90% 训练集, 5% 测试集, 5% 验证集
train_data, temp_data = train_test_split(filtered_data, test_size=0.1, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存训练集、测试集、验证集为 TSV 文件
train_data[['lhs', 'rel', 'rhs']].to_csv(SAVE_DIR+'/train.txt', sep='\t', index=False, header=False)
test_data[['lhs', 'rel', 'rhs']].to_csv(SAVE_DIR+'/test.txt', sep='\t', index=False, header=False)
val_data[['lhs', 'rel', 'rhs']].to_csv(SAVE_DIR+'/val.txt', sep='\t', index=False, header=False)
