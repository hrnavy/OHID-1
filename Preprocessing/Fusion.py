# -*- coding: <UTF-8> -*-
import numpy as np
import os
from osgeo import gdal
import time
import cv2


def write_gdal(im_data, path, im_proj=None, im_geotrans=None):
    '''
        重新写一个tiff图像
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param im_proj: 要设置的投影信息(默认None)
    :type im_proj: ?
    :param im_geotrans: 要设置的坐标信息(默认None)
    :type im_geotrans: ?
    :param path: 生成的图像路径(包括后缀名)
    :type path: string
    :return: None
    :rtype: None
    '''
    im_data = im_data.transpose((2, 0, 1))
    # print(im_data.dtype.name)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float32' in im_data.dtype.name:
        datatype = gdal.GDT_Float32
    else:
        datatype = gdal.GDT_Float64
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        if im_geotrans == None or im_proj == None:
            pass
        else:
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        print('writing the {}th band image'.format(i))
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def read_gdal(path):
    '''
        读取一个tiff图像
    :param path: 要读取的图像路径(包括后缀名)
    :type path: string
    :return im_data: 返回图像矩阵(h, w, c)
    :rtype im_data: numpy
    :return im_proj: 返回投影信息
    :rtype im_proj: ?
    :return im_geotrans: 返回坐标信息
    :rtype im_geotrans: ?
    '''
    image = gdal.Open(path)  # 打开该图像
    if image == None:
        print(path + "文件无法打开")
        return

    img_w = image.RasterXSize  # 栅格矩阵的列数
    img_h = image.RasterYSize  # 栅格矩阵的行数
    # im_bands = image.RasterCount  # 波段数
    im_proj = image.GetProjection()  # 获取投影信息
    im_geotrans = image.GetGeoTransform()  # 仿射矩阵
    im_data = image.ReadAsArray(0, 0, img_w, img_h)

    # 二值图一般是二维，需要添加一个维度
    if len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]
    im_data = im_data.transpose((1, 2, 0))
    return im_data, im_proj, im_geotrans


def generator(path, out_path, bands):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if len(bands) != 3:
        ext = '.tif'
    else:
        ext = '.tif'
    _files = os.listdir(out_path)
    _file = []
    if _files != None:
        for file in _files:
            # 这里将输出路径里面的文件名（不带后缀）全部读取出来
            _file.append(file.split('.')[0])

    # 读取path中的一个或多个高光谱原始影像
    # roots: 上一层文件夹的路径(str)
    # dirs: 该文件夹下所有子文件夹名称(list)
    # files: 该文件夹下的所有子文件名称(list)
    for roots, dirs, files in os.walk(path):
        # 如果该文件夹下没有文件，则说明是制作多个高光谱影像的波段合并
        if len(files) <= 0:
            for d in dirs:
                # 如果输出路径的不带后缀的文件名与输入路径的文件夹名称对应
                # 则继续循环，不执行fusion
                if d in _file:
                    print(d + "已存在于输出文件夹内")
                    continue
                # new_path为子文件夹的路径
                new_path = os.path.join(path, d)
                new_out = os.path.join(out_path, d + ext)

                # 执行fusion操作
                fusion(new_path, new_out, bands)
        # 如果该文件夹下有文件，则说明是制作单个高光谱影像的波段合并
        else:
            # 读取当前文件夹名称，其中path为当前文件夹路径
            dirname = os.path.basename(path)
            # 如果输出路径的不带后缀的文件名与输入路径的文件夹名称对应
            # 则继续循环，不执行fusion
            if dirname in _file:
                print(dirname + "已存在于输出文件夹内")
                continue
            new_out = os.path.join(out_path, dirname + ext)

            # 执行fusion操作
            fusion(path, new_out, bands)


def fusion(path, out_path, bands):
    print("执行fusion")
    start = time.time()
    files = os.listdir(path)
    Hyperspectral = np.zeros((5056, 5056, len(bands)), dtype=np.dtype('uint16'))

    bands_dict = {}
    for file in files:
        if file.split('.')[-1] in ['tif', 'TIF', 'tiff', 'TIFF']:
            # 根据文件名中的波段数字计算当前波段
            print(file)
            ind = int(file.split('_')[-2][1]) * 10 + int(file.split('_')[-2][2])
            # ind = int(file.split('.')[0].split('_')[-1])
            if ind in bands:
                bands_dict[ind] = os.path.join(path, file)

    for index, band in enumerate(bands):
        im_data, im_proj, im_geotrans = read_gdal(bands_dict[int(band)])

        if im_data.shape[0] > 5056:
            im_data = im_data[:5056, :]
        if im_data.shape[1] > 5056:
            im_data = im_data[:, :5056]
        Hyperspectral[:, :, index] = np.squeeze(im_data, axis=2)
        print('processing the {}th band image'.format(band), ' ' * 10 + 'channal:', index)

    write_gdal(Hyperspectral, out_path, im_proj, im_geotrans)
    end = time.time()
    print('Total time cost is {}'.format(end - start))


if __name__ == '__main__':
    '''
        path: 输入路径
        out_path: 输出路径
        
        输入说明：
        本代码存在批量合成模式和单合成模式：
        1.如果要对一个文件夹底下的所有子文件夹进行批量波段合成，则只需要填入该文件夹名称即可
            e.g. D:\\test文件夹下有 t1 文件夹和 t2 文件夹，批量合成只需在path中填入r'D:\test\'
        2.如果要对单个文件夹进行波段合成，只需填入该文件夹名称即可
            e.g. 对D:\\test\\t1 文件夹进行波段合成，只需在path中填入r'D:\test\t1'
            
        特别说明：
        代码逻辑自动判断文件夹下是否有子文件，如果有子文件，则默认使用单合成模式；
        如若只有子文件夹，没有任何文件，则使用批量合成模式。
    '''
    # 输入路径
    path = r'D:\OHS_data\firepix\HFM2_20200827222226_0001_L1B_CMOS3'

    # 输出路径
    out_path = r"D:\OHS_data\firepix\tif"

    # 波段数数组
    bands = []
    for i in range(32):
        bands.append(i + 1)

    # 真色彩波段（一般为[2,7,14],有时候也可以为[1，5，15]）
    # bands = [15, 5, 1]

    # 执行函数
    generator(path, out_path, bands)
