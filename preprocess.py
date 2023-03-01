from astropy.io import fits
import numpy as np
import phase3D2D
import math
import os


def minmax(array, pretreatment):
    """

    :param array:
    :param pretreatment:0归一化，1归一化后平方，2归一化后取log
    :return:
    """
    if pretreatment == 0:
        a_min = np.min(array)
        a_max = np.max(array)
        res = (array - a_min) / (a_max - a_min)
    elif pretreatment == 1:
        a_min = np.min(array)
        a_max = np.max(array)
        res = np.sqrt((array - a_min) / (a_max - a_min))
    elif pretreatment == 2:
        a_min = np.min(array)
        a_max = np.max(array)
        res = np.log((array - a_min) / (a_max - a_min))
    else:
        pass
    return res


def preprocess(filepath):

    sample_hdu = fits.open(filepath)
    image = np.stack((sample_hdu[2].data, sample_hdu[3].data)).astype(np.float32)
    image_in = image[0]
    image_in_path = 'D:/wfsExperiment/upload/show/image_in.png'
    image_in_normal_path = 'D:/wfsExperiment/upload/show/image_in_0.png'
    image_in_normal_sqrt_path = 'D:/wfsExperiment/upload/show/image_in_1.png'
    image_in_normal_log_path = 'D:/wfsExperiment/upload/show/image_in_2.png'

    web_image_in_path = 'wfsExperiment/upload/show/image_in.png'
    web_image_in_normal_path = 'wfsExperiment/upload/show/image_in_0.png'
    web_image_in_normal_sqrt_path = 'wfsExperiment/upload/show/image_in_1.png'
    web_image_in_normal_log_path = 'wfsExperiment/upload/show/image_in_2.png'
    if os.path.exists('D:/wfsExperiment/upload/show') == False:
        os.makedirs('D:/wfsExperiment/upload/show')
    print(image_in.shape)
    phase3D2D.phase_graph_3D(image_in, image_in_path)
    phase3D2D.phase_graph_3D(minmax(image_in, 0), image_in_normal_path)
    phase3D2D.phase_graph_3D(minmax(image_in, 1), image_in_normal_sqrt_path)
    phase3D2D.phase_graph_3D(minmax(image_in, 2), image_in_normal_log_path)

    return image_in_path, image_in_normal_path, image_in_normal_sqrt_path, image_in_normal_log_path
