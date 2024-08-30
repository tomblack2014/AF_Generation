import csv
import re
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 获取表头
        agent_names = header[1:]  # 获取agent名称列表

        for row in reader:
            agent_positions = re.findall(r'\(X=(.*?),Y=(.*?),Z=(.*?)\)', row[1])  # 提取位置信息
            data.append(agent_positions)

    return data


def downsample_data(data, factor):
    downsampled_data = []
    for agent_data in data:
        downsampled_agent_data = agent_data[::factor]
        downsampled_data.append(downsampled_agent_data)
    return downsampled_data


if __name__ == '__main__':
    data = read_csv_file('***.csv')

    # 降采样
    downsample_factor = 10  # 设置降采样因子
    downsampled_data = downsample_data(data, downsample_factor)

    finish_rate = 0
    count = 0
    # 绘制路径
    for agent_data in downsampled_data:
        x_values = []
        y_values = []

        # 解析位置信息
        for position in agent_data:
            x, y, _ = map(float, position)
            x_values.append(x)
            y_values.append(y)

        # 计算路径的瞬时方向，并根据方向确定颜色
        angles = []
        for i in range(1, len(agent_data)):
            x1, y1, _ = map(float, agent_data[i-1])
            x2, y2, _ = map(float, agent_data[i])
            angle = math.atan2(y2-y1, x2-x1)
            angles.append(angle)

        hsv = np.zeros((len(angles), 3))
        hsv[:, 0] = [(angle + math.pi) / (2 * math.pi) for angle in angles]
        hsv[:, 1] = 1
        hsv[:, 2] = 1

        colors = mcolors.hsv_to_rgb(hsv)

        plt.plot(x_values, y_values, color=colors[0])  # 使用第一个点的颜色作为整条路径的颜色
        for i in range(1, len(x_values)):
            plt.plot([x_values[i-1], x_values[i]], [y_values[i-1], y_values[i]], color=colors[i-1])  # 索引应减1

        count += 1
        finish_rate = count / len(data)
        print(finish_rate)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()