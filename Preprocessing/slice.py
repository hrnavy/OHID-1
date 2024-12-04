from tqdm import tqdm
from osgeo import gdal, ogr, osr
import numpy as np
from glob import glob
import os


def read_tif(tif_path):
    ds = gdal.Open(tif_path)
    row = ds.RasterXSize
    col = ds.RasterYSize
    band = ds.RasterCount

    for i in range(band):
        data = ds.GetRasterBand(i + 1).ReadAsArray()

        data = np.expand_dims(data, 2)
        if i == 0:
            allarrays = data
        else:
            allarrays = np.concatenate((allarrays, data), axis=2)
    return {'data': allarrays, 'transform': ds.GetGeoTransform(), 'projection': ds.GetProjection(), 'bands': band,
            'width': row, 'height': col}
    # 左上角点坐标 GeoTransform[0],GeoTransform[3] Transform[1] is the pixel width, and Transform[5] is the pixel height


def write_tif(fn_out, im_data, transform, proj=None):
    '''
    功能:
    ----------
    将矩阵按某种投影写入tif，需指定仿射变换矩阵，可选渲染为rgba

    参数:
    ----------
    fn_out:str
        输出tif图的绝对文件路径
    im_data: np.array
        tif图对应的矩阵
    transform: list/tuple
        gdal-like仿射变换矩阵，若im_data矩阵起始点为左上角且投影为4326，则为
            (lon_x.min(), delta_x, 0,
             lat_y.max(), 0, delta_y)
    proj: str（wkt格式）
        投影，默认投影坐标为4326，可用osr包将epsg转化为wkt格式，如
            srs = osr.SpatialReference()# establish encoding
            srs.ImportFromEPSG(4326)    # WGS84 lat/lon
            proj = srs.ExportToWkt()    # create wkt fromat of proj
    '''

    # 设置投影，proj为wkt format
    if proj is None:
        proj = 'GEOGCS["WGS 84",\
                     DATUM["WGS_1984",\
                             SPHEROID["WGS 84",6378137,298.257223563, \
                                    AUTHORITY["EPSG","7030"]], \
                             AUTHORITY["EPSG","6326"]], \
                     PRIMEM["Greenwich",0, \
                            AUTHORITY["EPSG","8901"]], \
                     UNIT["degree",0.0174532925199433, \
                            AUTHORITY["EPSG","9122"]],\
                     AUTHORITY["EPSG","4326"]]'
    # 渲染为rgba矩阵
    # 设置数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 将(通道数、高、宽)顺序调整为(高、宽、通道数)
    # print('shape of im data:', im_data.shape)
    im_bands = min(im_data.shape)
    im_shape = list(im_data.shape)
    im_shape.remove(im_bands)
    im_height, im_width = im_shape
    band_idx = im_data.shape.index(im_bands)
    # 找出波段是在第几个

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fn_out, im_width, im_height, im_bands, datatype)

    # if dataset is not None:
    dataset.SetGeoTransform(transform)  # 写入仿射变换参数
    dataset.SetProjection(proj)  # 写入投影

    if im_bands == 1:

        # print(im_data[:, 0,:].shape)
        if band_idx == 0:
            dataset.GetRasterBand(1).WriteArray(im_data[0, :, :])
        elif band_idx == 1:
            dataset.GetRasterBand(1).WriteArray(im_data[:, 0, :])
        elif band_idx == 2:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])

    else:

        for i in range(im_bands):
            if band_idx == 0:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i, :, :])
            elif band_idx == 1:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, i, :])
            elif band_idx == 2:
                dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])

    dataset.FlushCache()
    del dataset
    driver = None


def split(filename, origin_data, origin_transform, output_size):
    origin_size = origin_data.shape
    x = origin_transform[0]
    y = origin_transform[3]
    x_step = origin_transform[1]
    y_step = origin_transform[5]
    output_x_step = x_step
    output_y_step = y_step
    for i in range(origin_size[0] // output_size[0]):
        for j in range(origin_size[1] // output_size[1]):
            output_data = origin_data[i * output_size[0]:(i + 1) * output_size[0],
                          j * output_size[1]:(j + 1) * output_size[1], :]
            output_transform = (
                x + j * output_x_step * output_size[0], output_x_step, 0, y + i * output_y_step * output_size[0], 0,
                output_y_step)
            out_path = f'D:/fire512/' + filename + f'_{i}_{j}.tif'
            write_tif(out_path, output_data, output_transform)


import cv2
import os


def tianchong_you(img):
    size = img.shape
    # 这里的大小可以自己设定，但是尽量是32的倍数
    constant = cv2.copyMakeBorder(img, 0, 0, 0, 800 - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))  # 填充值为数据集均值
    return constant


def tianchong_xia(img):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, 800 - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(107, 113, 115))
    return constant


def tianchong_xy(img):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, 640 - size[0], 0, 640 - size[1], cv2.BORDER_CONSTANT, value=(107, 113, 115))
    return constant


def split_overlap(path, path_out, size_w=512, size_h=512, step=128):

    step = size_w - overlap
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    ims_list = []
    for file in os.listdir(path):
        end = file.split('.')[-1]
        if end in ['tif', 'tiff']:
            ims_list.append(file)
    count = 0
    for im_list in ims_list:
        number = 0
        name = im_list[:-4]  # 去除“.png后缀”
        print(name)
        img = cv2.imread(path + im_list)
        size = img.shape
        if size[0] >= size_h and size[1] >= size_w:
            count = count + 1
            for h in range(0, size[0] - 1, step):
                star_h = h
                for w in range(0, size[1] - 1, step):
                    star_w = w
                    end_h = star_h + size_h
                    if end_h > size[0]:
                        star_h = size[0] - size_h
                        end_h = star_h + size_h
                    end_w = star_w + size_w
                    if end_w > size[1]:
                        star_w = size[1] - size_w
                    end_w = star_w + size_w
                    cropped = img[star_h:end_h, star_w:end_w]
                    name_img = name + '_' + str(star_h) + '_' + str(star_w)  # 用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
                    cv2.imwrite('{}/{}.tif'.format(path_out, name_img), cropped)
                    number = number + 1
        if size[0] >= size_h and size[1] < size_w:
            print('图片{}需要在右面补齐'.format(name))
            count = count + 1
            img0 = tianchong_you(img)
            for h in range(0, size[0] - 1, step):
                star_h = h
                star_w = 0
                end_h = star_h + size_h
                if end_h > size[0]:
                    star_h = size[0] - size_h
                    end_h = star_h + size_h
                end_w = star_w + size_w
                cropped = img0[star_h:end_h, star_w:end_w]
                name_img = name + '_' + str(star_h) + '_' + str(star_w)
                cv2.imwrite('{}/{}.tif'.format(path_out, name_img), cropped)
                number = number + 1
        if size[0] < size_h and size[1] >= size_w:
            count = count + 1
            print('图片{}需要在下面补齐'.format(name))
            img0 = tianchong_xia(img)
            for w in range(0, size[1] - 1, step):
                star_h = 0
                star_w = w
                end_w = star_w + size_w
                if end_w > size[1]:
                    star_w = size[1] - size_w
                    end_w = star_w + size_w
                end_h = star_h + size_h
                cropped = img0[star_h:end_h, star_w:end_w]
                name_img = name + '_' + str(star_h) + '_' + str(star_w)
                cv2.imwrite('{}/{}.tif'.format(path_out, name_img), cropped)
                number = number + 1
        if size[0] < size_h and size[1] < size_w:
            count = count + 1
            print('图片{}需要在下面和右面补齐'.format(name))
            img0 = tianchong_xy(img)
            cropped = img0[0:size_h, 0:size_w]
            name_img = name + '_' + '0' + '_' + '0'
            cv2.imwrite('{}/{}.tif'.format(path_out, name_img), cropped)
            number = number + 1
        print('图片{}切割成{}张'.format(name, number))
        print('共完成{}张图片'.format(count))


if __name__ == '__main__':
    img_path = r'D:\OHS_data\firepix\tif16/'  # 图像数据集的路径
    out_path = r'D:\OHS_data\firepix\tif8'  # 切割得到的数据集存放路径
    size_h, size_w = 640, 640
    overlap = 0
    split_overlap(img_path, out_path, size_w, size_h, overlap)


