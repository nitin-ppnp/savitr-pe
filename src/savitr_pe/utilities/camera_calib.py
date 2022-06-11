import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as osp
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def extract_frames(calib_dir):

    os.mkdir(osp.join(calib_dir,"calib_images"))
    imdir = osp.join(calib_dir,"calib_images")

    vid_file = glob.glob(osp.join(calib_dir,"*.mp4"))[0]
    txt_file = glob.glob(osp.join(calib_dir,"*.txt"))[0]
    
    with(open(txt_file,"r")) as f:
        tstamps = f.readlines()
    
    vid_tstamp = vid_file.split("_")[-1].split(".")[0]
    tstamps = list(filter(None,tstamps[0].split(" ")))
    tstamps = np.array([int(x)-int(vid_tstamp) for x in tstamps])
    
    # read file
    cap = cv2.VideoCapture(vid_file)

    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idcs = np.round((tstamps/1000) * fps).astype(np.int)

    curr_frame_idx = 0
    # read and save frames
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            if curr_frame_idx in frame_idcs:
                cv2.imwrite(osp.join(imdir,"{:013d}".format(curr_frame_idx)+".jpg"),curr_frame)
            curr_frame_idx += 1
        else:
            break

    return imdir


def calibrate(dirpath, prefix, image_format, square_size, width=8, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)
    
    for fname in tqdm(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            cv2.imwrite(".".join(fname.split(".")[:-1])+"_calib."+fname.split(".")[-1],img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=True, help='image prefix')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size in meters')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 8')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()
    ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height)
    save_coefficients(mtx, dist, args.save_file)
    print("Calibration is finished. RMS: ", ret)