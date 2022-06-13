import glob
import sys
from os.path import join as ospj
import subprocess

dset_root = sys.argv[1]

im_dirs = sorted(glob.glob(ospj(dset_root,"*","VID*/")))
im_dirs = [x for x in im_dirs if ("pare" not in x and "pose" not in x)]

for j,dir in enumerate(im_dirs):
    subprocess.run(["python", "scripts/demo.py", "--image_folder", dir, "--output_folder", dir[:-1]+"_pare", "--no_render", 
    "--mode", "folder"])