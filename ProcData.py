import os
import glob

# 设置CSV文件所在目录
csv_dir = ''

# 获取所有CSV文件路径名
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# 初始化最小值和对应文件名
min_val = None
min_file = None

# 读取每个CSV文件
for csv_file in csv_files:
    with open(csv_file, 'r') as f:
        # 逐行读取CSV文件中的浮点数
        for line in f:
            val = float(line.strip())

            # 如果是第一个读取到的值或者比当前最小值还小
            if min_val is None or val < min_val:
                min_val = val
                min_file = csv_file

# 打印最小值和对应文件的路径名
print('Minimum value:', min_val)
print('Found in file:', min_file)