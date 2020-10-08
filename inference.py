import torch
import torch.nn as nn
import torchvision.transforms as T
from network2 import GwcNet
from read_data import ToTensor, Normalize, Pad, KITTI2015
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import cv2

max_disp = 192
left_path = '/home/jade/桌面/lc/gc-net/test/left/000001_10.png'
right_path = '/home/jade/桌面/lc/gc-net/test/right/000001_10.png'
model_path = 'test/model/kitti_gwc.ckpt'
save_path = '/home/jade/桌面/lc/gc-net/test/pic/disp_kitti.png'

mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
h=256
w=512
maxdisp=192

def main():
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    pairs = {'left': left , 'right' : right}
    transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    # transform = T.Compose([Normalize(mean, std), ToTensor()])
    pairs = transform(pairs)
    left = pairs['left'].to(device).unsqueeze(0)
    right = pairs['right'].to(device).unsqueeze(0)

    model = PSMNet(maxdisp).to(device)

    state = torch.load(model_path)
    if len(device_ids) == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        state['state_dict'] = new_state_dict

    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(model_path))
    # print('epoch: {}'.format(state['epoch']))
    # print('3px-error: {}%'.format(state['error']))

    model.eval()
    with torch.no_grad():
        _,_,disp = model(left, right)

    disp = disp.squeeze(0).detach().cpu().numpy()
    cv2.imwrite(save_path,disp)
    # plt.figure(figsize=(12.84, 3.84))
    # plt.axis('off')
    # plt.imshow(disp)
    # plt.colorbar()
    # plt.savefig(save_path, dpi=100)

    print('save diparity map in {}'.format(save_path))


def validate():
    validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    validate_dataset = KITTI2015('/home/jade/桌面/lc/KITTI', mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=1, num_workers=1)
    num_batchs = len(validate_loader)
    model = GwcNet(maxdisp).to(device)
    state = torch.load(model_path)

    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in state['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    state['state_dict'] = new_state_dict
    model.load_state_dict(state['state_dict'])
    print('load model from {}'.format(model_path))

    model.eval()
    avg_error = 0.0

    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = target_disp.gt(0)
        mask = mask.detach_()

        with torch.no_grad():
            disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])

        error_mat = ((delta >= 3.0) == 1)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100
        print('No.{:} : 3px-error: {:.5}%'.format(i,error))

        avg_error+=error

    avg_error = avg_error / num_batchs
    print('total : 3px-error : {:.5}%'.format(avg_error))

if __name__ == '__main__':
    # main()
    validate()