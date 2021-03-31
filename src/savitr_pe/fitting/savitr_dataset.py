import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import glob
import pickle as pkl
import json
import numpy as np


op_map2smpl = np.array([8,12,9,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
dlc_map2smpl = np.array([-1,3,2,-1,4,1,-1,5,0,-1,-1,-1,-1,-1,-1,-1,9,8,10,7,11,6,-1,-1])

class savitr_dataset(Dataset):
    def __init__(self,dir_path,NNs = ["alphapose","openpose"]):
        super().__init__()
        
        # Parse the dataset directory
        vid_files = glob.glob(os.path.join(dir_path,"*.mp4"))
        cams_dirs = [fl.split(".")[0] for fl in vid_files]
        self.num_cams = len(cams_dirs)
        
        # Image files dictionary with keys as timestamps
        self.cams_images_paths = []
        for cam_dir in cams_dirs:
            self.cams_images_paths.append(sorted(glob.glob(cam_dir+"/*")))
            # im_files = os.listdir(os.path.join(cams_dirs,"images"))
            # im_files_abs_paths = [os.path.abspath(f) for f in im_files]
            # im_files_dict = {t.split(".")[-1].split(".")[0] : t for t in im_files_abs_paths}
            # self.cams_images_paths.append(im_files_dict)

        # opose_m1 = pkl.load(open(osp.join(cams_dirs[0]+"_openpose.pkl"),"rb"))
        # opose_m2 = pkl.load(open(osp.join(cams_dirs[1]+"_openpose.pkl"),"rb"))
        apose_m1 = json.load(open(osp.join(cams_dirs[0]+"_alphapose","alphapose-results.json"),"r"))
        apose_m2 = json.load(open(osp.join(cams_dirs[1]+"_alphapose","alphapose-results.json"),"r"))
        
        self.apose_m1 = np.stack([np.reshape(apose_m1[x]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl] for x in sorted(apose_m1.keys())])
        self.apose_m2 = np.stack([np.reshape(apose_m2[x]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl] for x in sorted(apose_m2.keys())])
        self.apose_m1[:,al_map2smpl==-1,2] = 0
        self.apose_m2[:,al_map2smpl==-1,2] = 0

        self.m1_keys = np.array([k.split(".")[0] for k in sorted(apose_m1.keys())])
        self.m2_keys = np.array([k.split(".")[0] for k in sorted(apose_m2.keys())])
        self.frames_keys = np.array(sorted(np.union1d(self.m1_keys,self.m2_keys),key=int))
        self.frames2m1_map = np.where(np.isin(self.frames_keys,self.m1_keys))[0]
        self.frames2m2_map = np.where(np.isin(self.frames_keys,self.m2_keys))[0]

        # check if 2D joints exits
        for cam in cams_dirs:
            for nn in NNs:
                if not os.path.exists(os.path.join(cam,"_",nn)):
                    # 2D joints processing if 2d joints does not exists
                    pass
                    # save the 2D joints file

        # Read the joints and camera parameters in a dictionary
        # with keys as timestamp

        # Create union for timeline

        # Create indexed list of closest timestamps from each views

    def __len__(self):
        pass


    def __getitem__(self, index):
        pass

