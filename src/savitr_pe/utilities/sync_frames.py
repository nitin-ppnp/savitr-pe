import numpy as np
from pydub import AudioSegment
import cv2
import os
import os.path as osp
import pandas as pd

def sync_audio(file0,file1,start_idx=[0,0],end_idx=[-1,-1]):
    '''
    file0: first input file
    file1: second input file
    return: offset of second track w.r.t first track
    '''

    # read files
    audio0 = AudioSegment.from_file(file0)
    audio1 = AudioSegment.from_file(file1)
    
    # sampling rates should be same
    assert audio0.frame_rate == audio1.frame_rate

    # get samples
    w0 = np.array(audio0.get_array_of_samples())[start_idx[0]:end_idx[0]]/1e9
    w1 = np.array(audio1.get_array_of_samples())[start_idx[1]:end_idx[1]]/1e9

    # correlation
    corr = np.correlate(w0,w1,"full")

    # get offset in ms
    offset = 1000*(np.argmax(corr) - w1.shape[0] - start_idx[0] + start_idx[1])/audio0.frame_rate


    return offset


def frame_extract_and_sync(vid_file,outdir,time_offset=0):
    '''
    vid_file: input video file
    outdir: target directory for saving frames
    time_offset: offset to be applied to frame timestamps (in ns)
    '''
    # read file
    cap = cv2.VideoCapture(vid_file)

    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create out directory
    os.mkdir(outdir)

    tstamp_file = ".".join(vid_file.split(".")[:-1])+"_timestamps.csv"

    # read and save frames
    if osp.exists(tstamp_file):
        tstamps = pd.read_csv(tstamp_file, header=None, usecols=[0]).to_numpy()
        frame_num = 0
        while(cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                cv2.imwrite(os.path.join(outdir,"{:015d}".format(int(tstamps[frame_num])+time_offset)+".jpg"),curr_frame)
                frame_num += 1
            else:
                break
    else:
        while(cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                cv2.imwrite(os.path.join(outdir,"{:015d}".format(int(cap.get(cv2.CAP_PROP_POS_MSEC)*1000000+time_offset))+".jpg"),curr_frame)
            else:
                break

    cap.release()
