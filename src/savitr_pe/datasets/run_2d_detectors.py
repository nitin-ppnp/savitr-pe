import glob
import sys
from os.path import join as ospj
import subprocess

dset_root = sys.argv[1]

im_dirs = sorted(glob.glob(ospj(dset_root,"*","VID*/")))
im_dirs = [x for x in im_dirs if ("pare" not in x and "pose" not in x)]


for j,dir in enumerate(im_dirs):
    subprocess.run(["singularity", "exec", "--nv", "-B", "/is:/is", "-B", "/ps:/ps",
    "-B", "/home/nsaini:/home/nsaini", 
    "/is/ps3/nsaini/projects/openpose_scripts/savitr.simg", "python3", 
    "/is/ps3/nsaini/projects/savitr_pe/src/scripts/openpose_script_modified.py", 
    "--input_dir", dir, 
    "--pkl_path", dir[:-1]+"_openpose.pkl", 
    "--output_dir", dir[:-1]+"_openpose"])

    # subprocess.run(["singularity", "exec", "--nv", "-B", "/is:/is",  "-B", "/ps:/ps",
    # "-B", "/home/nsaini:/home/nsaini", 
    # "/is/ps3/nsaini/projects/openpose_scripts/opose_sandbox", "python3", "/AlphaPose/demo.py", 
    # "--indir", dir, 
    # "--outdir", dir[:-1]+"_alphapose", 
    # "--save_img", "--format", "open"])