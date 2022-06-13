from savitr_pe.utilities import sync_frames
import numpy as np
import glob
import os
import os.path as osp
from pydub import AudioSegment


def prepare_dataset(dset_path,start_idx=None, end_idx=None):

    # sync files
    mp4_files = sorted(glob.glob(dset_path+"/*/*.mp4"))
    t_offset = {}
    for i in range(len(mp4_files)):
        if start_idx is None:
            start = [0,0]
        else:
            start = [start_idx[0],start_idx[i]]
        if end_idx is None:
            end = [-1,-1]
        else:
            end = [end_idx[0],end_idx[i]]
        t_offset[str(mp4_files[i].split("/")[-2])] = sync_frames.sync_audio(mp4_files[0],mp4_files[i],start_idx=start,end_idx=end)
    t_offset[str(mp4_files[0].split("/")[-2])] = 0

    # extract frames
    for i in range(len(mp4_files)):
        sync_frames.frame_extract_and_sync(mp4_files[i],mp4_files[i].split(".")[0])

    # save offset file
    if not osp.exists(osp.join(dset_path,"t_offset")):
        np.savez(osp.join(dset_path,"t_offset"),**t_offset)
    else:
        print("Offset file already exists")

if __name__ == "__main__":
    dset_path = "/home/nsaini/Datasets/opencam/VolleyDay"
    prepare_dataset(dset_path,start_idx=[900000,900000,900000,900000],end_idx=[2100000,2100000,2100000,2100000])
    # prepare_dataset(dset_path)