from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import cv2


def pfm_png(pfm_file_path, png_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channel = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().strip())
        if scale < 0:
            endian = '<'  # little endlian
            scale = -scale
        else:
            endian = '>'  # big endlian

        disparity = np.fromfile(pfm_file, endian + 'f')

        img = np.reshape(disparity, newshape=(height, width))
        img = np.flipud(img)
        # plt.imsave(os.path.join(png_file_path), img)
        cv2.imwrite(os.path.join(png_file_path), img)

def main():
    pfm_file_dir = 'D:/lc/data/disparity/TRAIN/15mm_focallength/scene_backwards/fast/left/0001.pfm'
    png_file_dir = 'D:/lc/gc-net/test/pic.png'
    pfm_png(pfm_file_dir, png_file_dir)


if __name__ == '__main__':
    main()