from urllib.parse import uses_params
import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import glob
import pickle as pkl
import json
import numpy as np
import pandas as pd
import pytorch3d.transforms as p3dt
from ..utilities import camera_calib
from scipy.integrate import cumtrapz
import joblib
from scipy.spatial.transform import Rotation


op_map2smpl = np.array([8,-1,-1,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
dlc_map2smpl = np.array([-1,3,2,-1,4,1,-1,5,0,-1,-1,-1,-1,-1,-1,-1,9,8,10,7,11,6,-1,-1])

class savitr_dataset(Dataset):
    def __init__(self,dir_path,NNs = ["alphapose","openpose"], seq_len=None, use_sensor_data=False):
        super().__init__()
        
        # Parse the dataset directory
        self.cams_dirs = sorted(glob.glob(dir_path+"/*/"))
        vid_files = [glob.glob(os.path.join(x,"*.mp4"))[0] for x in self.cams_dirs]
        self.num_cams = len(self.cams_dirs)
        self.cam_names = [x.split("/")[-1].split(".")[0] for x in vid_files]
        self.im_dirs = [osp.join(self.cams_dirs[i],self.cam_names[i]) for i in range(len(self.cams_dirs))]

        # sequence length to get from the dataset
        if seq_len is None:
            self.seq_len = 1
        else:
            self.seq_len = seq_len
        
        # read time offset
        audio_offsets_fl = np.load(osp.join(dir_path,"t_offset.npz"))
        self.audio_offsets = [np.round(audio_offsets_fl[x.split("/")[-2]]*1000000) for x in self.cams_dirs]   # first offset should be 0
        
        
        # Image files dictionary with keys as timestamps
        self.cams_images_paths = []
        for im_dir in self.im_dirs:
            self.cams_images_paths.append(sorted(glob.glob(im_dir+"/*")))

        # apose_res = [json.load(open(osp.join(self.cams_dirs[i],self.cam_names[i] + "_alphapose","alphapose-results.json"),"r")) for i in range(len(self.cams_dirs))]
        # self.apose_res = []
        # for i in range(len(apose_res)):
        #     self.apose_res.append({})
        #     for x in sorted(apose_res[i].keys()):
        #         apose = np.reshape(apose_res[i][x]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
        #         apose[al_map2smpl==-1,:] = 0
        #         self.apose_res[i][x] = apose
        
        opose_res = [pkl.load(open(osp.join(self.cams_dirs[i],self.cam_names[i] + "_openpose.pkl"),"rb")) for i in range(len(self.cams_dirs))]

        self.opose_res = []
        for i in range(len(opose_res)):
            self.opose_res.append({})
            for x in sorted(opose_res[i].keys()):
                opose_pose = opose_res[i][x]["pose"]
                if opose_pose.shape == ():
                    opose = np.zeros((24,3))
                else:
                    # get the 2d pose with biggest bounding box 
                    person_id = np.stack([opose_pose[i,opose_pose[i,:,2] != 0,:2].max(axis=0) - 
                            opose_pose[i,opose_pose[i,:,2] != 0,:2].min(axis=0) for i in range(opose_pose.shape[0])]).sum(1).argmax()
                    opose = opose_pose[person_id,op_map2smpl]
                opose[op_map2smpl==-1,:] = 0
                self.opose_res[i][x] = opose
        
        self.tstamps = [np.array(sorted(self.opose_res[i].keys())) for i in range(len(self.opose_res))]

        # calculate clock offsets
        self.start_offsets = [int(self.tstamps[i][0]) - int(self.tstamps[0][0]) for i in range(len(self.tstamps))]

        int_tstamps_offset_corr = [np.array(list(map(int,self.opose_res[i].keys()))).astype(np.long) + self.audio_offsets[i] - self.start_offsets[i] for i in range(len(self.opose_res))]

        # intersection of timeline
        tstamp_start = max([int_tstamps_offset_corr[i][0] for i in range(len(int_tstamps_offset_corr))])
        tstamp_end = min([int_tstamps_offset_corr[i][-1] for i in range(len(int_tstamps_offset_corr))])
        isect_idcs = [np.arange(len(int_tstamps_offset_corr[i]))[(int_tstamps_offset_corr[i] >= tstamp_start) & (int_tstamps_offset_corr[i] <= tstamp_end)] for i in range(len(int_tstamps_offset_corr))]

        # create final tstamps
        self.tstamps_final = [self.tstamps[0][isect_idcs[0]]]
        for i in range(1,len(int_tstamps_offset_corr)):
            closest_idcs = np.array([(np.abs(int_tstamps_offset_corr[i]-x)).argmin() for x in int_tstamps_offset_corr[0][isect_idcs[0]]])
            self.tstamps_final.append(self.tstamps[i][closest_idcs])
            

        # get sensor data
        if use_sensor_data:
            self.rot_data = []
            self.accel_data = []
            self.pos_data = []
            for i in range(len(int_tstamps_offset_corr)):
                rot_df = pd.read_csv(osp.join(self.cams_dirs[i],self.cam_names[i]+"rotation.csv")).to_numpy()
                closest_idcs = np.array([(np.abs(rot_df[:,5].astype(np.int64)-x)).argmin() for x in np.int64(self.tstamps_final[i])])
                rot_data = p3dt.quaternion_to_matrix(torch.from_numpy(rot_df[closest_idcs,:4][:,[3,0,1,2]]).float())
                self.rot_data.append(rot_data.detach().cpu().numpy())

            for i in range(len(int_tstamps_offset_corr)):
                accel_df = pd.read_csv(osp.join(self.cams_dirs[i],self.cam_names[i]+"accel.csv")).to_numpy()
                gravity_df = pd.read_csv(osp.join(self.cams_dirs[i],self.cam_names[i]+"gravity.csv")).to_numpy()
                closest_idcs = np.array([(np.abs(gravity_df[:,3].astype(np.int64)-x)).argmin() for x in np.int64(accel_df[:,3])])
                accel_data = accel_df[closest_idcs,:3] - gravity_df[closest_idcs,:3]
                pos_data = cumtrapz(cumtrapz(accel_data,x=accel_df[:,3]/1e9,axis=0,initial=0),x=accel_df[:,3]/1e9,axis=0,initial=0)
                closest_idcs = np.array([(np.abs(accel_df[:,3].astype(np.int64)-x)).argmin() for x in np.int64(self.tstamps_final[i])])
                self.pos_data.append(pos_data[closest_idcs,:])
                self.accel_data.append(accel_data[closest_idcs,:])


        # get intrinsics
        self.cam_intr = []
        for c in range(self.num_cams):
            calib_params_file = glob.glob(osp.join(self.cams_dirs[c],"*.yml"))
            if len(calib_params_file) != 1:
                import ipdb;ipdb.set_trace()
            self.cam_intr.append(camera_calib.load_coefficients(calib_params_file[0])[0])

        self.cam_intr = np.stack(self.cam_intr)


        # if PARE results exists
        use_pare = True
        for c in range(self.num_cams):
            if not osp.exists(osp.join(self.cams_dirs[c],self.cam_names[c])):
                use_pare = False
                
        
        
        

    def __len__(self):
        self.tstamps_final[0].shape[0] - self.seq_len + 1


    def __getitem__(self, index, seq_len=None):
        
        # if seq_len is not specified, use single frame
        if seq_len is None:
            seq_len = self.seq_len
        if seq_len is None:
            seq_len = 1

        # assert index + seq_len should be withing the dataset
        assert index + seq_len <= self.tstamps_final[0].shape[0]


        # get full images paths
        full_img_pth = []
        for cam in range(self.num_cams):
            full_img_pth_cam = [osp.join(self.im_dirs[cam],"{}.jpg".format(self.tstamps_final[cam][index+x])) for x in range(seq_len)]
            full_img_pth.append(full_img_pth_cam)
        full_img_pth = np.array(full_img_pth)

        # get 2d keypoints
        j2d = []
        for cam in range(self.num_cams):   
            j2d.append(torch.stack([torch.from_numpy(self.opose_res[cam][self.tstamps_final[cam][index+x]]).float() for x in range(seq_len)]))
        j2d = torch.stack(j2d)

        full_img_pth_list = [list(x) for x in list(full_img_pth)]

        # get pare results
        pare_res = []
        pare_orient = []
        pare_cams = []
        for x in range(seq_len):
            pred_pose = []
            pred_cam = []
            for c in range(self.num_cams):
                pare_file = osp.join(self.cams_dirs[c],
                                        self.cam_names[c]+"_pare",self.cam_names[c]+"_",
                                        "pare_results",self.tstamps_final[c][index+x]+".pkl")
                if osp.exists(pare_file):
                    pare_file_out = joblib.load(pare_file)
                    pred_pose.append(pare_file_out["pred_pose"][0][:22])
                    pred_cam.append(pare_file_out["pred_cam_t"][0])
                elif x == 0:
                    pred_pose.append(np.eye(3)[np.newaxis].repeat(22,axis=0))
                    pred_cam.append(np.array([0,0,30]).astype(np.float32))
                else:
                    # print("pare pose not found for cam {}".format(c))
                    pred_pose.append(pred_pose_last[c])
                    pred_cam.append(pred_cam_last[c])
            
            pred_pose_last = np.stack(pred_pose)
            pred_cam_last = np.stack(pred_cam)
            pare_res.append(np.stack([Rotation.from_matrix(pred_pose_last[:,i,:3,:3]).mean().as_rotvec() for i in range(1,22)]))
            pare_orient.append(Rotation.from_matrix(pred_pose_last[:,0,:3,:3]).as_rotvec())
            pare_cams.append(pred_cam_last)

        pare_cams = torch.from_numpy(np.stack(pare_cams)[0] * (self.cam_intr[:, 0, 0:1] + self.cam_intr[:, 1, 1:2]) / (2 * 5000)).unsqueeze(0).float()
        pare_pose = torch.from_numpy(np.stack(pare_res)).float()
        pare_orient = torch.from_numpy(np.stack(pare_orient,axis=1)).float()

        return {"full_im_paths":full_img_pth_list, "j2d":j2d, "cam_intr":self.cam_intr, 
                "pare_poses":pare_pose, "pare_orient":pare_orient,
                "pare_cams":pare_cams}

        # return {"full_im_paths":full_img_pth_list, "j2d":j2d, "cam_intr":self.cam_intr}






