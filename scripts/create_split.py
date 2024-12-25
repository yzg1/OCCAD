import os
import random

# 设置随机数种子
random.seed(42)

# 设置源数据目录和输出目录
source_dir = './data/renderingimg/edgemap'
output_dir = './data/train_val_test_list'

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 获取所有文件
all_files = [f for f in os.listdir(source_dir) if f.isdigit()]

# 打乱文件顺序
random.shuffle(all_files)

# 计算划分的索引
total_files = len(all_files)
train_size = int(0.8 * total_files)
val_size = int(0.1 * total_files)

# 划分数据集
train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]

# 输出文件路径
train_file = os.path.join(output_dir, 'train.txt')
val_file = os.path.join(output_dir, 'val.txt')
test_file = os.path.join(output_dir, 'test.txt')

# 将文件名写入到文本文件
def write_to_txt(file_list, output_file):
    with open(output_file, 'w') as f:
        for file_name in file_list:
            f.write(f"{file_name}\n")

write_to_txt(train_files, train_file)
write_to_txt(val_files, val_file)
write_to_txt(test_files, test_file)

print(f"train.txt is saved to {train_file}\nval.txt is saved to {val_file}\ntest.txt is saved to {test_file}")
