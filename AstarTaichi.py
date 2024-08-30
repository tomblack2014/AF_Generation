import taichi as ti
import numpy as np

node = ti.types.struct(g=ti.float32, h=ti.float32, f=ti.float32, father=ti.int32)


@ti.func
def check_in_list(field: ti.template(), element: ti.i32, len: ti.i32) -> ti.i8:
    """
    检查某一个元素是否存在于field列表中 返回1存在
    len: 希望检查的长度
    """
    flag = 0
    for i in range(len):
        if flag == 1:
            continue
        if field[i] == element:
            flag = 1
    return flag


@ti.func
def normalize(vec):
    """向量标准化"""
    norm = vec.norm()
    new_vec = ti.Vector([0.0, 0.0])
    if norm != 0:
        new_vec = vec / norm
    return new_vec


@ti.data_oriented
class AStar:
    def __init__(self, filename):
        np_map = self.init_from_filename(filename)
        self.window_size = np_map.shape[0]
        self.pixel_number = int(self.window_size * self.window_size)

        self.result = ti.Vector.field(2, dtype=ti.f32)  # 记录每个像素格子到达目标地点的A*单位方向向量
        ti.root.dense(ti.i, self.window_size).dense(ti.j, self.window_size).place(self.result)

        self.map = ti.field(ti.f32, shape=(self.window_size, self.window_size))
        self.map.from_numpy(np_map)

        _list_offset = np.array([(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)])
        self.list_offset = ti.Vector.field(2, ti.i8, shape=8)
        self.list_offset.from_numpy(_list_offset)
        self.dist_list_offset = ti.field(ti.f32, shape=8)
        self.init_list_offset()

        self.node_matrix = node.field(shape=self.pixel_number)  # 创建存储节点的矩阵
        self.open_list = ti.field(ti.i32)  # 初始化打开列表 待遍历的节点编号
        self.close_list = ti.field(ti.i32)  # 初始化关闭列表 已经遍历过的节点编号
        ti.root.pointer(ti.i, self.pixel_number).place(self.open_list)
        ti.root.pointer(ti.i, self.pixel_number).place(self.close_list)

    def init_from_filename(self, filename):
        map_data = []
        with open(filename, 'r') as file:
            for line in file:
                row = []
                for char in line.strip():
                    if char.isdigit():
                        row.append(int(char))
                map_data.append(row)
        np_map = np.array(map_data)
        return np_map

    @ti.kernel
    def init_list_offset(self):
        """A*计算时对相邻网格的扩展"""
        for i in self.list_offset:
            self.dist_list_offset[i] = ((self.list_offset[i][0] ** 2 + self.list_offset[i][1] ** 2) * 100) ** 0.5

    def AstarCal(self, startpos, targetpos):
        target = targetpos[0] + self.window_size * targetpos[1]
        start = startpos[0] + self.window_size * startpos[1]

        if self.map[targetpos[0], targetpos[1]] == 1 or start == target:  # 如果在障碍列表，则跳过,起始地点=目标地点则直接返回
            return (0, 0)

        # 对重复使用的数据结构做初始化（taichi kernel中不支持定义临时的field数据结构）
        self.node_matrix.fill(0)
        self.open_list.fill(0)
        self.close_list.fill(0)
        self.result.fill(0)

        self.next_loc(start, target)

    @ti.func
    def check_out_map(self, index_i, index_j):
        flag = False
        if index_i < 0 or index_i >= self.window_size or index_j < 0 or index_j >= self.window_size:
            flag = True
        return flag

    @ti.kernel
    def next_loc(self, start_pos: ti.i32, target_pos: ti.i32):
        """
        输入:起始地点的index,目标地点的index
        index计算方法为: index = x * window_size + y,(x,y为坐标,如[350,50],window_size为网格边长,假定为正方形网格)
        使用A*计算起始地点到目标地点的最短路径
        算出最短路之后,记录路径上所有节点到目标地点的方向向量到self.result中
        """
        self.open_list[0] = start_pos  # 起始点添加至打开列表
        open_list_len = 1  # 初始化打开列表 待遍历的节点
        close_list_len = 0  # 初始化关闭列表 已经遍历过的节点
        target_x = ti.floor(target_pos / self.window_size)
        target_y = target_pos % self.window_size

        # 开始算法的循环
        while True:
            #  判断是否停止 若目标节点在关闭列表中则停止循环
            if check_in_list(self.close_list, target_pos, close_list_len) == 1:
                break

            now_loc = self.open_list[0]
            place = 0
            #   （1）获取f值最小的点
            for i in range(0, open_list_len):
                if self.node_matrix[self.open_list[i]].f < self.node_matrix[now_loc].f:
                    now_loc = self.open_list[i]
                    place = i
                    #   （2）切换到关闭列表
            open_list_len += -1
            self.open_list[place] = self.open_list[open_list_len]
            self.close_list[close_list_len] = now_loc
            close_list_len += 1

            grid_x = int(now_loc / self.window_size)
            grid_y = now_loc % self.window_size
            for i in range(8):  # （3）对3*3相邻格中的每一个
                index_i = grid_x + self.list_offset[i][0]
                index_j = grid_y + self.list_offset[i][1]
                if self.check_out_map(index_i, index_j):  # 越界直接跳过
                    continue
                if self.map[int(index_i), int(index_j)] == 1:  # 如果在障碍列表，则跳过
                    continue

                # 1214: 障碍物只能贴边走
                if i > 3 and self.map[int(grid_x + self.list_offset[i][0]), grid_y] == 1:
                    continue
                if i > 3 and self.map[grid_x, int(self.list_offset[i][1] + grid_y)] == 1:
                    continue

                index = int(index_i * self.window_size + index_j)
                if check_in_list(self.close_list, index, close_list_len):  # 如果在关闭列表，则跳过
                    continue

                #  该节点不在open列表，添加，并计算出各种值
                if not check_in_list(self.open_list, index, open_list_len):
                    self.open_list[open_list_len] = index
                    open_list_len += 1

                    self.node_matrix[index].g = self.node_matrix[now_loc].g + self.dist_list_offset[i]
                    self.node_matrix[index].h = (abs(target_x - index_i) + abs(target_y - index_j)) * 10  # 采用曼哈顿距离
                    self.node_matrix[index].f = (self.node_matrix[index].g + self.node_matrix[index].h)
                    self.node_matrix[index].father = now_loc
                    continue
                #  如果在open列表中，比较，重新计算
                if self.node_matrix[index].g > self.node_matrix[index].g + self.dist_list_offset[i]:
                    self.node_matrix[index].g = self.node_matrix[index].g + self.dist_list_offset[i]
                    self.node_matrix[index].father = now_loc
                    self.node_matrix[index].f = (self.node_matrix[index].g + self.node_matrix[index].h)

        #  找到最短路 依次遍历父节点，找到下一个位置 close列表中的father都可以复用
        next_move = target_pos
        current = self.node_matrix[next_move].father

        while next_move != start_pos:
            index_i = int(next_move / self.window_size)
            index_j = next_move % self.window_size
            i = int(current / self.window_size)
            j = int(current % self.window_size)
            #  记录去往下一个位置的标准化方向向量
            re = ti.Vector([int(index_i - i), int(index_j - j)])
            self.result[i, j] = normalize(re)
            next_move = current
            current = self.node_matrix[next_move].father
