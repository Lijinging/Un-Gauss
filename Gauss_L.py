# coding:utf-8
# 一维高斯模糊 -xlxw
from PIL import Image as p
import numpy as np
from time import clock
import math
import os

# define
sizepic = [0, 0]
timer = [0, 0, 0, 0]
PI = math.pi


def getrgb(path, r):  # 得到图像中各个点像素的通道值
    timer[0] = clock()
    pd = p.open(path).convert("L")
    sizepic[0], sizepic[1] = pd.size[0], pd.size[1]
    n_p = np.zeros((sizepic[0], sizepic[1]))
    for i in range(0, sizepic[0]):
        for j in range(0, sizepic[1]):
            n_p[i][j] = pd.getpixel((i, j))
    # 镜像扩充
    for i in range(1, r + 1):  # 顶部
        nx_p = n_p[i * 2 - 1]
        n_p = np.insert(n_p, 0, values=nx_p, axis=0)
    for i in range(sizepic[0] + r - 1, sizepic[0] - 1, -1):  # 底部
        nx_p = n_p[i]
        n_p = np.insert(n_p, (sizepic[0] + r - 1) * 2 - i, values=nx_p, axis=0)
    for i in range(1, r + 1):  # 左侧
        nx_p = n_p[:, i * 2 - 1]
        n_p = np.insert(n_p, 0, values=nx_p, axis=1)
    for i in range(sizepic[1] + r - 1, sizepic[1] - 1, -1):  # 右侧
        nx_p = n_p[:, i]
        n_p = np.insert(n_p, (sizepic[1] + r - 1) * 2 - i, values=nx_p, axis=1)
    print("已经得到所有像素值，所花时间为{:.3f}s".format(clock() - timer[0]))
    timer[0] = clock() - timer[0]
    return n_p


def matcombine(n_p, rd):
    # 模糊矩阵
    summat = 0
    timer[1] = clock()
    ma = np.zeros(2 * rd + 1)
    for i in range(0, 2 * rd + 1):
        ma[i] = (1 / (((2 * PI) ** 0.5) * rd)) * math.e ** (-((i - rd) ** 2) / (2 * (rd ** 2)))
        summat += ma[i]
    ma[0::1] /= summat
    print("已经计算出高斯函数矩阵，所花时间为{:.3f}s".format(clock() - timer[1]))
    timer[1] = clock() - timer[1]
    # blur
    ne_p = np.zeros_like(n_p)
    u, p, q = 0, 0, 0
    # y向模糊
    timer[2] = clock()

    for i in range(rd + 1, sizepic[0] + rd - 1):
        for j in range(rd + 1, sizepic[1] + rd - 1):
            u += n_p[j - rd:j + rd + 1:1, i] * ma[0::1]
            ne_p[j][i] = u.sum(0)
            u = 0
    # x向模糊
    for i in range(rd + 1, sizepic[0] + rd - 1):
        for j in range(rd + 1, sizepic[1] + rd - 1):
            u += ne_p[i, j - rd:j + rd + 1:1] * ma[0::1]
            ne_p[i][j] = u.sum(0)
            u = 0
    print("已经完成模糊，所花时间为{:.3f}s".format(clock() - timer[2]))
    timer[2] = clock() - timer[2]
    return ne_p


def cpic(n_p, path, rd):  # 图片输出
    timer[3] = clock()
    pd = p.new("L", (sizepic[0] - rd - 1, sizepic[1] - rd - 1))
    for i in range(rd + 1, sizepic[0]):
        for j in range(rd + 1, sizepic[1]):
            pd.putpixel((i - rd - 1, j - rd - 1), (int(n_p[i][j])))
    print("已经完成生成，所花时间为{:.3f}s".format(clock() - timer[3]))
    timer[3] = clock() - timer[3]
    print("正在导出图片..")
    pd.save(path)


def main():
    rd = eval(input("请输入模糊的半径："))
    path = input("请输入图片的地址(包括后缀)：")
    nr, ng, nb = getrgb(path, rd)
    nr, ng, nb = matcombine(nr, ng, nb, rd)
    cpic(nr, ng, nb, path, rd)
    print("{} - >> {}".format(path.split('/')[-1], "blurred.jpg"))
    print("总计耗时:{:.3f}s,感谢您的使用.".format(timer[0] + timer[1] + timer[2] + timer[3]))


if __name__ == '__main__':
    rd = 8
    path = r"IMG_RAW/"
    outpath = r"IMG_GAUSS/"
    filelist = os.listdir(path)
    print(filelist)

    for infile in filelist:
        filepath = path + infile
        print(filepath)
        n_p = getrgb(filepath, rd)
        n_p = matcombine(n_p, rd)
        cpic(n_p, outpath + infile, rd)


    pass

