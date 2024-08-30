import taichi as ti
import numpy as np
import Astar
import pygame
import random
import struct
import matplotlib.pyplot as plt

def replace_suffix(s, old_suffix, new_suffix):
    # 如果原后缀是字符串的一部分，则替换它
    if s.endswith(old_suffix):
        return s[:-len(old_suffix)] + new_suffix
    return s

def read_map_from_file(filename):
    map_data = []
    with open(filename, 'r') as file:
        for line in file:
            row = []
            for char in line.strip():
                if char.isdigit():
                    row.append(int(char))
            map_data.append(row)
    return map_data


def generate_random_position(map_data):
    rows = len(map_data)
    cols = len(map_data[0])
    while True:
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        if map_data[row][col] == 0:
            return [col, row]

def cal_distance(v):
    return v[0] * v[0] + v[1] * v[1]

#filename = ["map_5.txt"]
filename = []
size = [0.1, 0.02, 0.05, 0.15, 0.2, 0.25, 0.3]
dateStr = "0502\\2024-05-02_"
for i in range(7):
    for j in range(5):
        filename.append(dateStr + "50_50_" + str(size[i]) + "_2\\map_" + str(j + 1) + ".txt")

for index in range(35):
    ti.init(ti.cpu)  # 用Astar.py之前一定要先init taichi
    myastar = Astar.AStar(filename[index])# 传文件名
    map_data = read_map_from_file(filename[index])

    filename[index] = replace_suffix(filename[index], ".txt", ".jpg")

    # 进行数次astar搜索以生成AF
    astar_times = 200000
    rows = len(map_data)
    cols = len(map_data[0])
    af_data = np.zeros((rows, cols, 8), dtype=float)

    for num in range(astar_times):
        start = generate_random_position(map_data)
        end = generate_random_position(map_data)
        myastar.AstarCal(start, end)# 指定startpos，endpos
        np_result = myastar.result.to_numpy()#结果在myastar.result这个taichi field里 可以直接转numpy
        for i in range(rows):
            for j in range(cols):
                if cal_distance(np_result[i][j]) > 0:
                    dirs = [[1, 0], [0.707, 0.707], [0, 1], [-0.707, 0.707], [-1, 0], [-0.707, -0.707], [0, -1], [0.707, -0.707]]
                    for k in range(8):
                        v = [np_result[i][j][0] - dirs[k][0], np_result[i][j][1] - dirs[k][1]]
                        if cal_distance(v) < 0.1:
                            af_data[i][j][k] = af_data[i][j][k] + 1
        if num % 10000 == 0:
            print(num)
    # np.set_printoptions(threshold=np.inf)

    # 对af中的值进行归一化
    for i in range(rows):
        for j in range(cols):
            sum = 0
            for k in range(8):
                sum += af_data[i][j][k]
            if sum < 1e-3:
                sum = 1
            for k in range(8):
                af_data[i][j][k] = af_data[i][j][k] / sum

    # 筛选af_data，找出其中方差大于阈值thres的，其他都直接清零
    thres = 0.04
    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            tMask = np.var(af_data[i][j]) > thres
            if not tMask:
                for k in range(8):
                    af_data[i][j][k] = 0
            mask[i][j] = tMask

    # 保存为图片
    mask_image = mask.astype(np.uint8) * 255  # 将True转换为255，False保持为0
    plt.imsave(filename[index], mask_image, cmap='gray')  # 使用灰度图来保存

    print("保存图片：" + filename[index])
    filename[index] = replace_suffix(filename[index], ".jpg", ".dst")

    # 可视化mask
    #plt.imshow(mask, cmap='gray', interpolation='nearest')
    #plt.colorbar()  # 显示颜色条（虽然在这个案例中可能不太有用，因为只用了黑白）
    #plt.title('Mask where af values are less than threshold')
    #plt.axis('off')  # 关闭坐标轴
    #plt.show()

    flattened_array = af_data.ravel(order='C')
    int_array = flattened_array.astype(np.float32)
    # 计算展开后数组的字节数
    num_bytes = len(flattened_array)
    packed_data = struct.pack(f'<{num_bytes}f', *int_array)

    packed_data_head = struct.pack('ii', rows, cols)

    with open(filename[index], 'wb') as f:
    #with open("af_data.dst", 'wb') as f:
        f.write(packed_data_head)
        f.write(packed_data)
    f.close()

    print("写入完成：" + filename[index])

# pygame.init()
# screen = pygame.display.set_mode((1000, 1000))
# pygame.display.set_caption("人群仿真环境")
#
# while True:
#     # 画map
#     screen.fill((255, 255, 255))
#     for i in range(100):
#         for j in range(100):
#             color = (0, 0, 0) if map_data[i][j] == 1 else (255, 255, 255)
#             pygame.draw.rect(screen, color, (j * 10, i * 10, 10, 10))
#
#     # 画af
#     for i in range(100):
#         for j in range(100):
#             if map_data[i][j] == 1:
#                 continue
#             val = np.max(af_data[i][j])
#             index = np.argmax(af_data[i][j])
#             colorlist = [[255,0,0], [0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[128,255,64],[64,128,255],[255,64,128]]
#             if val <= 0:
#                 pygame.draw.rect(screen, (255, 255, 255), (j * 10, i * 10, 10, 10))
#             else:
#                 pygame.draw.rect(screen, colorlist[index], (j * 10, i * 10, 10, 10))
#     pygame.display.flip()
