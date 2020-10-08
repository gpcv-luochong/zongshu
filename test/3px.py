import torch
import torch.nn.functional as F
import cv2
import numpy as np

disp_image = 'C:/Users/chong/Pictures/SGM/disp_whu.png'
disp_gt_image = 'C:/Users/chong/Pictures/SGM/pic/pic_whu.png'

device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))

def main():
    disp = cv2.imread(disp_image, cv2.IMREAD_ANYDEPTH)
    disp_gt = cv2.imread(disp_gt_image,0)

    # disp8U = np.zeros((disp.shape[0], disp.shape[1]))
    # disp8U=cv2.normalize(disp_gt, disp_gt, 0, 255, cv2.NORM_MINMAX)

    disp_gt = torch.from_numpy(disp_gt).type(torch.FloatTensor)
    disp = torch.from_numpy(disp).type(torch.FloatTensor)

    # disp_gt = F.pad(disp_gt.unsqueeze(0),pad=(0,6,0,9)).squeeze()
    # disp_gt = F.pad(disp_gt.unsqueeze(0), pad=(0, 0, 0, 4)).squeeze()

    mask = disp.gt(0)
    mask = mask.detach_()

    delta = torch.abs(disp[mask] - disp_gt[mask])
    # error_mat = (((delta >= 3.0) + (delta >= 0.05 * (disp_gt[mask]))) == 2)
    error_mat = ((delta >= 3.0) == 1)
    error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100

    print('3px-error: {:.5}%'.format(error))

if __name__=='__main__':
    main()