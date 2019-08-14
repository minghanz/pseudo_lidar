#!/usr/bin/env python
# Created by Minghan
import skimage
import skimage.io
import numpy as np
import preprocessing.kitti_util as kitti_util

from psmnet.pred_disp_module import DisparityNet
from preprocessing.generate_lidar_module import PclGenerator

import argparse
import os

def init_args(descp='PSMNet'):
    if descp == "PSMNet":
        parser = argparse.ArgumentParser(description='PSMNet')
        parser.add_argument('--KITTI', default='2015',
                            help='KITTI version')
        parser.add_argument('--datapath', default='/mnt/storage/minghanz_data/KITTI/object/testing/',
                            help='select model')
        parser.add_argument('--loadmodel', default='./psmnet/kitti_3d/finetune_300.tar',
                            help='loading model')
        parser.add_argument('--model', default='stackhourglass',
                            help='select model')
        parser.add_argument('--maxdisp', type=int, default=192,
                            help='maxium disparity')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--save_path', type=str, default='/mnt/storage/minghanz_data/KITTI/object/testing/predict_disparity', metavar='S',
                            help='path to save the predict')
        parser.add_argument('--save_figure', action='store_true', help='if true, save the numpy file, not the png file')
        args = parser.parse_args()
    elif descp == 'Generate Lidar':
        parser = argparse.ArgumentParser(description='Generate Lidar')
        parser.add_argument('--calib_dir', type=str,
                            default='/mnt/storage/minghanz_data/KITTI/object/testing/calib')
        parser.add_argument('--disparity_dir', type=str,
                            default='/mnt/storage/minghanz_data/KITTI/object/testing/predicted_disparity')
        parser.add_argument('--save_dir', type=str,
                            default='/mnt/storage/minghanz_data/KITTI/object/testing/predicted_velodyne')
        parser.add_argument('--max_high', type=int, default=1)
        args = parser.parse_args()

    return args

class detection3dPseudeLidarNode(object):
    def __init__(self):
        self.args_disp=init_args('PSMNet')
        self.args_gen_lidar = init_args('Generate Lidar')

        # network initialization
        self.disp_pred_net = DisparityNet(self.args_disp)
        self.pcl_generator = PclGenerator(self.args_gen_lidar)
        
    def run_from_file(self):
        if self.args_disp.KITTI == '2015':
            from psmnet.dataloader import KITTI_submission_loader as DA
        else:
            from psmnet.dataloader import KITTI_submission_loader2012 as DA  
        test_left_img, test_right_img = DA.dataloader(self.args_disp.datapath)
    
        if not os.path.isdir(self.args_disp.save_path):
            os.makedirs(self.args_disp.save_path)

    
        for inx in range(len(test_left_img)):
            ################ part 1 #######################
            imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
            imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
    
            img = self.disp_pred_net.run(imgL_o, imgR_o)

            # # file output
            # print(test_left_img[inx].split('/')[-1])
            # if self.args.save_figure:
            #     skimage.io.imsave(self.args.save_path+'/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
            # else:
            #     np.save(self.args.save_path+'/'+test_left_img[inx].split('/')[-1][:-4], img)

            ################# part 2 ###################
            predix = test_left_img[inx].split('/')[-1][:-4]
            calib_file = '{}/{}.txt'.format(self.args_gen_lidar.calib_dir, predix)
            calib = kitti_util.Calibration(calib_file)

            img = (img*256).astype(np.uint16)/256.
            lidar = self.pcl_generator.run(calib, img)

            # pad 1 in the indensity dimension
            lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
            lidar = lidar.astype(np.float32)
            lidar.tofile('{}/{}.bin'.format(self.args_gen_lidar.save_dir, predix))
            print('Finish Depth {}'.format(predix))

            ##################### part 3 ############################
            
      
def main():
    detection_node = detection3dPseudeLidarNode()
    detection_node.run_from_file()

if __name__ == "__main__":
    main()