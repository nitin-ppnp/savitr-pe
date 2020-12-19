import numpy as np
from pydub import AudioSegment

def sync_frames(file0,file1):
    '''
    file0: first input file
    file1: second input file
    return: offset of second track w.r.t first track
    '''

    # read files
    audio0 = AudioSegment.from_file(file0)
    audio1 = AudioSegment.from_file(file1)
    
    # get samples
    w0 = np.array(audio0.get_array_of_samples())/1e9
    w1 = np.array(audio1.get_array_of_samples())/1e9

    # correlation
    corr = np.correlate(w0,w1,"full")

    # get offset
    offset = np.argmax(corr) - w1.shape[0]

    return offset
