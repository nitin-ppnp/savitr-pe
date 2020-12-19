import torch
from torch.utils.data import Dataset
import os

class savitr_dataset(Dataset):
    def __init__(self,dir_path,NNs = ["alphapose","openpose"]):
        super().__init__()
        
        # Parse the dataset directory
        cams_dirs = os.listdir(dir_path)
        self.num_cams = len(cams_dirs) 

        
        # Image files dictionary with keys as timestamps
        self.cams_images_paths = []
        for cam_dir in cams_dirs:
            im_files = os.listdir(os.path.join(cams_dir,"images"))
            im_files_abs_paths = [os.path.abspath(f) for f in im_files]
            im_files_dict = {t.split(".")[-1].split(".")[0] : t for t in im_files_abs_paths}
            self.cams_images_paths.append(im_files_dict)

        
        # check if 2D joints exits
        for cam in cams_dirs:
            for nn in NNs:
                if not os.path.exists(os.path.join(cam,nn+"results")):
                    # 2D joints processing if 2d joints does not exists

                    # save the 2D joints file

        # Read the joints and camera parameters in a dictionary
        # with keys as timestamp

        # Create intersection for timeline

        # Create indexed list of closest timestamps from each views

    def __len__(self):


    def __getitem__(self, index):


