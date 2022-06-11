# %%
import ipdb;ipdb.set_trace()
import torch
import torchvision
from savitr_pe.fitting import savitr_dataset
from savitr_pe.utilities.renderer import Renderer
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel
from savitr_pe.utilities.geometry import perspective_projection,transform_smpl
from pytorch3d import transforms
import pickle as pkl
from tqdm import tqdm
from torch import autograd
import cv2
import os
import numpy as np
import copy
from savitr_pe.utilities import camera_calib

device = torch.device("cuda")
smpl2op_jmap = torch.tensor([15,12,17,19,21,16,18,20,2,5,8,1,4,7])
cmap = np.random.rand(14,3)
# %% bad grad check code
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)
    
    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return torch.isnan(grad_output).any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

def kp_viz(im,j2d):
    im_cp = copy.deepcopy(im)
    for i,j in enumerate(j2d[smpl2op_jmap]):
        cv2.circle(im_cp,(j[0],j[1]),10,cmap[i],-1)
    return im_cp

# %% get the datset

ds = savitr_dataset.savitr_dataset("/is/ps3/nsaini/projects/savitr_pe/media/nitin_priyanka_1616103206968_gNNpkTrv92JWHUDGCNJS3b")

smplx_model = BodyModel(bm_path="/is/ps3/nsaini/projects/copenet/src/copenet/data/smplx/models/smplx/SMPLX_NEUTRAL.npz")
smplx_model.to(device)
smplx_model.eval()

vp_model = load_model("/ps/scratch/common/vposer/V02_05", model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]
vp_model.to(device)
vp_model.eval()

# def geman mcclure
def gmcclure(a,b,sigma=30):
    x = a-b
    return x**2/(x**2+sigma**2)

# %% Initial parameters
batch_size = ds.frames_keys.shape[0]
batch_size_m1 = len(ds.frames2m1_map)
batch_size_m2 = len(ds.frames2m2_map)
im0_res = np.array(cv2.imread(ds.cams_images_paths[0][0]).shape[1::-1])
im1_res = np.array(cv2.imread(ds.cams_images_paths[1][0]).shape[1::-1])

pl_smplxbeta = torch.zeros(10,requires_grad=True).float().to(device).detach().clone()
pl_smplxz = torch.zeros(batch_size,32,requires_grad=True).float().to(device).detach().clone()
pl_smplxphi = torch.eye(3,requires_grad=True).float().expand(batch_size,-1,-1).to(device).detach().clone()
pl_smplxtau = torch.tensor([0.,0.,3.],requires_grad=True).float().expand(batch_size,-1).to(device).detach().clone()
pl_intr0 = torch.from_numpy(camera_calib.load_coefficients("/is/ps3/nsaini/projects/savitr_pe/media/cv_calib_priyanka_1617022377792")[0]).float().to(device)
pl_intr1 = torch.from_numpy(camera_calib.load_coefficients("/is/ps3/nsaini/projects/savitr_pe/media/cv_calib_nitin_1617022071348")[0]).float().to(device)
pl_extr0 = torch.eye(4).float().to(device)
pl_rot1 = torch.eye(3,requires_grad=True).float().to(device).detach().clone()
pl_trans1 = torch.zeros(3,requires_grad=True).float().to(device).detach().clone()
pl_extr1 = torch.cat([torch.cat([pl_rot1,pl_trans1.unsqueeze(1)],dim=1),torch.tensor([0,0,0,1]).to(device).float().unsqueeze(0)],dim=0)

pl_intr0[:2,2] = torch.from_numpy(im0_res//2).to(device)
pl_intr1[:2,2] = torch.from_numpy(im1_res//2).to(device)
pl_intr0[0,0] /= 5
pl_intr0[1,1] /= 5
pl_intr1[0,0] /= 5
pl_intr1[1,1] /= 5

apose_m1 = torch.from_numpy(ds.apose_m1).float().to(device)
apose_m2 = torch.from_numpy(ds.apose_m2).float().to(device)

loss_fun = torch.nn.MSELoss(reduction="none")         
apose_m1[:,[1,2],2] /= 2   # less weight for hips
apose_m1[:,[1,2],2] /= 2

w_beta = 2000
w_vposer = 100
w_temporal = 0

pl_smplxz.requires_grad = True
pl_smplxphi.requires_grad = True
pl_smplxtau.requires_grad = True
pl_smplxbeta.requires_grad = True
# pl_rot1.requires_grad = True
# pl_trans1.requires_grad = True

optim = torch.optim.Adam([pl_smplxz,
                                pl_smplxphi,
                                pl_smplxtau,
                                pl_smplxbeta],
                                lr=0.01)
renderer0 = Renderer(focal_length=[pl_intr0[0,0],pl_intr0[1,1]], 
                                img_res=np.array(im0_res),
                                center=pl_intr0[:2,2],
                                faces=smplx_model.f.data.cpu())
renderer1 = Renderer(focal_length=[pl_intr1[0,0],pl_intr1[1,1]], 
                                img_res=np.array(im1_res),
                                center=pl_intr1[:2,2],
                                faces=smplx_model.f.data.cpu())
for iter in tqdm(range(1000)):

    image0 = cv2.imread(ds.cams_images_paths[0][0])/255.
    image1 = cv2.imread(ds.cams_images_paths[1][0])/255.
    
    pl_smplxtheta = vp_model.decode(pl_smplxz)["pose_body"].reshape(-1,63)
    smplx_out = smplx_model.forward(betas=pl_smplxbeta.unsqueeze(0).expand([batch_size,-1]), 
                                        pose_body=pl_smplxtheta,
                                        root_orient=torch.zeros(batch_size,3,device=device).float(),
                                        trans = torch.zeros(batch_size,3,device=device).float())

    transf_mat = torch.cat([pl_smplxphi[:,:3,:3],
                                pl_smplxtau.unsqueeze(2)],dim=2)
    verts,joints3d,_,_ = transform_smpl(transf_mat,
                                    smplx_out.v,
                                    smplx_out.Jtr)
    joints3d0 = joints3d[ds.frames2m1_map]
    joints3d1 = joints3d[ds.frames2m2_map]
    
    joints2d0 = perspective_projection(joints3d0[:,:24],
                                                    rotation=pl_extr0[:3,:3].unsqueeze(0).expand([batch_size_m1,-1,-1]),
                                                    translation=pl_extr0[:3,3].expand([batch_size_m1,-1]),
                                                    focal_length=[pl_intr0[0,0],pl_intr0[1,1]],
                                                    camera_center=pl_intr0[:2,2]).squeeze(0)
    joints2d1 = perspective_projection(joints3d1[:,:24],
                                                    rotation=pl_rot1.unsqueeze(0).expand([batch_size_m2,-1,-1]),
                                                    translation=pl_trans1.unsqueeze(0).expand([batch_size_m2,-1]),
                                                    focal_length=[pl_intr1[0,0],pl_intr1[1,1]],
                                                    camera_center=pl_intr1[:2,2]).squeeze(0)
    # import ipdb;ipdb.set_trace()
    loss_2d = (apose_m1[:,:,2:]*loss_fun(joints2d0,apose_m1[:,:,:2])).mean() #+ \
                # (apose_m2[:,:,2:]*loss_fun(joints2d1,apose_m2[:,:,:2])).mean()

    loss_vposer = torch.mul(pl_smplxz,pl_smplxz).mean()

    loss_beta = torch.mul(pl_smplxbeta,pl_smplxbeta).mean()
    
    loss_temporal = loss_fun(pl_smplxtheta[1:],pl_smplxtheta[:-1]).mean() + \
                    loss_fun(pl_smplxphi[1:],pl_smplxphi[:-1]).mean() + \
                    loss_fun(pl_smplxtau[1:],pl_smplxtau[:-1]).mean()

    loss = loss_2d + w_beta*loss_beta + w_vposer*loss_vposer + w_temporal*loss_temporal

    
    if iter % 100 == 0:
        # viz #############################
        vis0 = renderer0(verts[1].detach().clone().cpu(),
                                        pl_extr0[:3,3].detach().clone().cpu(),
                                        pl_extr0[:3,:3].detach().clone().cpu(),
                                        image0)
        vis1 = renderer1(verts[0].detach().clone().cpu(),
                                        pl_trans1.detach().clone().cpu(),
                                        pl_rot1.detach().clone().cpu(),
                                        image1)
        # vis0 = kp_viz(vis0,joints2d0[0])
        # vis1 = kp_viz(vis1,joints2d1[0])
        
        for i in range(24):
            cv2.circle(vis0,tuple(joints2d0[i,:2].detach().cpu().numpy()),10,(255,255,255),-1)
            cv2.circle(vis1,tuple(joints2d1[i,:2].detach().cpu().numpy()),10,(255,255,255),-1)
        
        cv2.imshow("im0",vis0[::3,::3])
        cv2.imshow("im1",vis1[::3,::3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # msh = Mesh(v=verts[0].detach().cpu().numpy(),f=smplx_model.faces)
        # mv.static_meshes = [msh]
    
        # for m in [0,end-1]:
        #     vis["mesh"+str(m)].set_object(g.TriangularMeshGeometry(verts[m].detach().cpu().numpy(),smplx_model.faces))
        ##################################

    
    optim.zero_grad()
    loss.backward()
    optim.step()

import ipdb;ipdb.set_trace()

for begin in tqdm([0,2000,4000,6000]):

    if begin == 6000:
        end = 6990
    else:
        end = begin + 2000
    lseq = end-begin

    sigma2d = 40
    
    w_beta = 2000
    w_vposer = 0.05
    w_temporal = 0
    intr0 = torch.from_numpy(ds.intr0).float().to(device)
    intr1 = torch.from_numpy(ds.intr1).float().to(device)

    renderer0 = Renderer(focal_length=[intr0[0,0],intr0[1,1]], 
                                img_res=[1920,1080],
                                center=intr0[:2,2],
                                faces=smplx_model.f.data.cpu())
    renderer1 = Renderer(focal_length=[intr1[0,0],intr1[1,1]], 
                        img_res=[1920,1080],
                        center=intr1[:2,2],
                        faces=smplx_model.f.data.cpu())

    # %% all the vairables
    cam1_extr = torch.eye(4,device=device).float()
    smplxbeta = torch.zeros([10],device=device).float().clone()

    # fix parameters
    cam0_extr = torch.eye(4,device=device).float()

    sub_err_idcs = err_idcs[begin:end]
    sub_robust_idcs = robust_idcs[begin:end]
    
    # viz

    # from psbody.mesh.meshviewer import MeshViewer
    # from psbody.mesh.mesh import Mesh
    # mv = MeshViewer()

    # import meshcat
    # import meshcat.geometry as g
    # import meshcat.transformations as tf
    # vis = meshcat.Visualizer()

    # main loop
    with autograd.detect_anomaly():
        pl_smplxtheta = smpl_z_init[begin:end].detach().clone()
        pl_smplxtheta.requires_grad = True
        pl_smplxphi0 = transforms.matrix_to_rotation_6d(smpl_rotmat0[begin:end,0,:3,:3]).detach().clone()
        pl_smplxphi0.requires_grad = True
        pl_smplxtau0 = smpl_wrt_cam0[begin:end,:3,3].detach().clone()
        pl_smplxtau0.requires_grad = True
        pl_smplxphi1 = transforms.matrix_to_rotation_6d(smpl_wrt_cam1[begin:end,:3,:3] ).detach().clone()
        pl_smplxphi1.requires_grad = True
        pl_smplxtau1 = smpl_wrt_cam1[begin:end,:3,3].detach().clone()
        pl_smplxtau1.requires_grad = True
        pl_smplxbeta = smplxbeta.detach().clone()
        pl_smplxbeta.requires_grad = True
        n_iters = 300

        # create optimizer
        optim1 = torch.optim.Adam([pl_smplxphi0,
                                pl_smplxtau0,
                                pl_smplxphi1,
                                pl_smplxtau1,
                                pl_smplxbeta],
                                lr=0.01)
        optim2 = torch.optim.Adam([pl_smplxtheta,
                                pl_smplxphi0,
                                pl_smplxtau0,
                                pl_smplxphi1,
                                pl_smplxtau1,
                                pl_smplxbeta],
                                lr=0.01)
        optim = optim1
        
        image0 = cv2.imread(ds.get_j2d_only(begin)["im0_path"])/255.
        image1 = cv2.imread(ds.get_j2d_only(begin)["im1_path"])/255.
        global verts0
        global verts1
        global pl_smplxphi0_9d
        global pl_smplxphi1_9d

        for j in tqdm(range(n_iters)):
            
            if j == 100:
                optim = optim2
            
            pl_smplxtheta_3d = vp_model.decode(pl_smplxtheta)["pose_body"].reshape(-1,63)
            
            # forward SMPLX
            smplx_out = smplx_model.forward(betas=pl_smplxbeta.unsqueeze(0).expand([lseq,-1]), 
                                        pose_body=pl_smplxtheta_3d,
                                        root_orient=torch.zeros(lseq,3,device=device).float(),
                                        trans = torch.zeros(lseq,3,device=device).float())
            pl_smplxphi0_9d = transforms.rotation_6d_to_matrix(pl_smplxphi0).squeeze(0)
            pl_smplxphi1_9d = transforms.rotation_6d_to_matrix(pl_smplxphi1).squeeze(0)
            
            transf_mat0 = torch.cat([pl_smplxphi0_9d[:,:3,:3],
                                pl_smplxtau0.unsqueeze(2)],dim=2)
            transf_mat1 = torch.cat([pl_smplxphi1_9d[:,:3,:3],
                                pl_smplxtau1.unsqueeze(2)],dim=2)
            verts0,joints3d0,_,_ = transform_smpl(transf_mat0,
                                    smplx_out.v,
                                    smplx_out.Jtr)
            verts1,joints3d1,_,_ = transform_smpl(transf_mat1,
                                    smplx_out.v,
                                    smplx_out.Jtr)

            # joints3d0 = torch.matmul(j_regressor,verts0)
            # joints3d1 = torch.matmul(j_regressor,verts1)
            joints2d0 = perspective_projection(joints3d0[:,:24],
                                                    rotation=cam0_extr[:3,:3].unsqueeze(0).expand([lseq,-1,-1]),
                                                    translation=cam0_extr[:3,3].expand([lseq,-1]),
                                                    focal_length=[intr0[0,0],intr0[1,1]],
                                                    camera_center=intr0[:2,2]).squeeze(0)
            
            joints2d1 = perspective_projection(joints3d1[:,:24],
                                                    rotation=cam1_extr[:3,:3].unsqueeze(0).expand([lseq,-1,-1]),
                                                    translation=cam1_extr[:3,3].expand([lseq,-1]),
                                                    focal_length=[intr1[0,0],intr1[1,1]],
                                                    camera_center=intr1[:2,2]).squeeze(0)

            sigma = sigma2d
            loss_fun = torch.nn.MSELoss(reduction="none")
            
            joints2d_gt0[:,:,[1,2],2:] /= 2   # less weight for hips
            joints2d_gt1[:,:,[1,2],2:] /= 2
            
            loss_2d = (joints2d_gt0[begin:end][sub_robust_idcs,0,:,2:]*gmcclure(joints2d0[sub_robust_idcs],joints2d_gt0[begin:end][sub_robust_idcs,0,:,:2])).mean() + \
                        (joints2d_gt1[begin:end][sub_robust_idcs,0,:,2:]*gmcclure(joints2d1[sub_robust_idcs],joints2d_gt1[begin:end][sub_robust_idcs,0,:,:2])).mean() + \
                        (joints2d_gt0[begin:end][sub_robust_idcs,1,:,2:]*gmcclure(joints2d0[sub_robust_idcs],joints2d_gt0[begin:end][sub_robust_idcs,1,:,:2])).mean() + \
                        (joints2d_gt1[begin:end][sub_robust_idcs,1,:,2:]*gmcclure(joints2d1[sub_robust_idcs],joints2d_gt1[begin:end][sub_robust_idcs,1,:,:2])).mean()
            # loss_2d =     (joints2d_gt0[begin:end][:,2,:,2:]*loss_fun(joints2d0[:],joints2d_gt0[begin:end][:,2,:,:2])).mean() + \
            #             (joints2d_gt1[begin:end][:,2,:,2:]*loss_fun(joints2d1[:],joints2d_gt1[begin:end][:,2,:,:2])).mean()

            

            loss_vposer = torch.mul(pl_smplxtheta,pl_smplxtheta).mean()

            loss_beta = torch.mul(smplxbeta,smplxbeta).mean()
            
            
            sub_robust_idcs_temporal = np.logical_and(sub_robust_idcs[:-1],sub_robust_idcs[1:])
            loss_temporal = 10*loss_fun(pl_smplxtheta_3d[1:],pl_smplxtheta_3d[:-1])[sub_robust_idcs_temporal].mean() + \
                            100*loss_fun(pl_smplxphi0[1:],pl_smplxphi0[:-1])[sub_robust_idcs_temporal].mean() + \
                            100*loss_fun(pl_smplxphi1[1:],pl_smplxphi1[:-1])[sub_robust_idcs_temporal].mean() + \
                            100*loss_fun(pl_smplxtau0[1:],pl_smplxtau0[:-1])[sub_robust_idcs_temporal].mean() + \
                            100*loss_fun(pl_smplxtau1[1:],pl_smplxtau1[:-1])[sub_robust_idcs_temporal].mean()   

            loss = loss_2d + w_beta*loss_beta + w_vposer*loss_vposer + w_temporal*loss_temporal
            

            # if j % 100 == 0:
            # # viz #############################
            #     vis0 = renderer0(verts0[0].detach().clone().cpu(),
            #                                     cam0_extr[:3,3].detach().clone().cpu(),
            #                                     cam0_extr[:3,:3].detach().clone().cpu(),
            #                                     image0)
            #     vis1 = renderer1(verts1[0].detach().clone().cpu(),
            #                                     cam1_extr[:3,3].detach().clone().cpu(),
            #                                     cam1_extr[:3,:3].detach().clone().cpu(),
            #                                     image1)
            #     # vis0 = kp_viz(vis0,joints2d0[0])
            #     # vis1 = kp_viz(vis1,joints2d1[0])
            #     cv2.imshow("im",np.concatenate([vis0[::3,::3],vis1[::3,::3]],axis=0))
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
                
                # msh = Mesh(v=verts[0].detach().cpu().numpy(),f=smplx_model.faces)
                # mv.static_meshes = [msh]
            
                # for m in [0,end-1]:
                #     vis["mesh"+str(m)].set_object(g.TriangularMeshGeometry(verts[m].detach().cpu().numpy(),smplx_model.faces))
            ###################################
            # import ipdb;ipdb.set_trace()
            # get_dot = register_hooks(loss)
            # loss.backward()
            # dot = get_dot()
            
            # print(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

    ########################################################
    pl_smpl_z[begin:end,:] = pl_smplxtheta.detach().clone()
    pl_smpl_wrt_cam0[begin:end,:3,:3] = pl_smplxphi0_9d
    pl_smpl_wrt_cam0[begin:end,:3,3] = pl_smplxtau0

    pl_smpl_wrt_cam1[begin:end,:3,:3] = pl_smplxphi1_9d
    pl_smpl_wrt_cam1[begin:end,:3,3] = pl_smplxtau1
    pl_cam1_wrt_cam0[begin:end] = torch.matmul(pl_smpl_wrt_cam0[begin:end],torch.inverse(pl_smpl_wrt_cam1[begin:end]))
    pl_verts0[begin:end] = verts0.detach().cpu().numpy()
    pl_verts1[begin:end] = verts1.detach().cpu().numpy()
    ########################################################

# %% baseline and airpose results load
images0 = np.array(ds.db["im0"])
images1 = np.array(ds.db["im1"])

airpose_pred_vertices_cam0 = torch.cat([i["output"]["pred_vertices_cam0"].to("cuda") for i in res[res_id]])
airpose_pred_vertices_cam1 = torch.cat([i["output"]["pred_vertices_cam1"].to("cuda") for i in res[res_id]])

fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/hmr/version_2_from_newlytrinedckpt/checkpoints/epoch=388.pkl"
fname0 = fname + "0"
fname1 = fname + "1"
res0 = pkl.load(open(fname0,"rb"))
res1 = pkl.load(open(fname1,"rb"))
bl_pred_vertices_cam0 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res0[res_id]])
bl_pred_vertices_cam1 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res1[res_id]])

for smpls in tqdm(range(2000,6990,5)):
    samples = [smpls,smpls+1,smpls+2,smpls+3,smpls+4]
    ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
    ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)
    crp0 = [ds[sample]["crop_info0"].detach().cpu().numpy() for sample in samples]
    crp1 = [ds[sample]["crop_info1"].detach().cpu().numpy() for sample in samples]

    bl_rend_ims0 = renderer0.visualize_tb(bl_pred_vertices_cam0[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0,color=(0.3,0.3,0.8,1))
    bl_rend_ims1 = renderer1.visualize_tb(bl_pred_vertices_cam1[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims1,color=(0.3,0.3,0.8,1))
    airpose_rend_ims0 = renderer0.visualize_tb(airpose_pred_vertices_cam0[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0,color=(0.8,0.3,0.8,1))
    airpose_rend_ims1 = renderer1.visualize_tb(airpose_pred_vertices_cam1[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims1,color=(0.8,0.3,0.8,1))
    aplus_rend_ims0 = renderer0.visualize_tb(torch.from_numpy(pl_verts0[samples]),
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0,color=(0.3,0.8,0.8,1))
    aplus_rend_ims1 = renderer1.visualize_tb(torch.from_numpy(pl_verts1[samples]),
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims1,color=(0.3,0.8,0.8,1))
    bl_rend_ims0 = torch.cat([bl_rend_ims0[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])
    bl_rend_ims1 = torch.cat([bl_rend_ims1[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])
    airpose_rend_ims0 = torch.cat([airpose_rend_ims0[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])
    airpose_rend_ims1 = torch.cat([airpose_rend_ims1[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])
    aplus_rend_ims0 = torch.cat([aplus_rend_ims0[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])
    aplus_rend_ims1 = torch.cat([aplus_rend_ims1[:,:,i*1920:(i+1)*1920].unsqueeze(0) for i in range(5)])

    crp_imgs0 = [torchvision.utils.make_grid(torch.cat([bl_rend_ims0[i,:,crp0[i][0,0]:crp0[i][1,0],crp0[i][0,1]:crp0[i][1,1]].unsqueeze(0),
                airpose_rend_ims0[i,:,crp0[i][0,0]:crp0[i][1,0],crp0[i][0,1]:crp0[i][1,1]].unsqueeze(0),
                aplus_rend_ims0[i,:,crp0[i][0,0]:crp0[i][1,0],crp0[i][0,1]:crp0[i][1,1]].unsqueeze(0)]),padding=3).permute(1,2,0) for i in range(len(samples))]
    crp_imgs1 = [torchvision.utils.make_grid(torch.cat([bl_rend_ims1[i,:,crp1[i][0,0]:crp1[i][1,0],crp1[i][0,1]:crp1[i][1,1]].unsqueeze(0),
                airpose_rend_ims1[i,:,crp1[i][0,0]:crp1[i][1,0],crp1[i][0,1]:crp1[i][1,1]].unsqueeze(0),
                aplus_rend_ims1[i,:,crp1[i][0,0]:crp1[i][1,0],crp1[i][0,1]:crp1[i][1,1]].unsqueeze(0)]),padding=3).permute(1,2,0) for i in range(len(samples))]
    for id,sample in enumerate(samples):
        cv2.imwrite(os.path.join("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/iccv21_res_imgs",dataset,"c0",str(sample)+".jpg"),crp_imgs0[id].detach().cpu().numpy()[:,:,::-1]*255)
    for id,sample in enumerate(samples):
        cv2.imwrite(os.path.join("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/iccv21_res_imgs",dataset,"c1",str(sample)+".jpg"),crp_imgs1[id].detach().cpu().numpy()[:,:,::-1]*255)
    import ipdb;ipdb.set_trace()
            # import meshcat
            # import meshcat.geometry as g
            # import meshcat.transformations as tf
            # vis = meshcat.Visualizer()
            # import time
            # for m in range(verts0.shape[0]):
            #     vis["mesh"].set_object(g.TriangularMeshGeometry(verts0[m].detach().cpu().numpy(),smplx_model.f.detach().cpu().numpy()))
            #     time.sleep(0.05)

            # camera_extr_opt[:,i] = pl_camera_extr.detach().clone()
            # smplxtheta[i] = pl_smplxtheta.detach().clone()
            # smplxphitau[i] = pl_smplxphitau.detach().clone()
            # smplxbeta[i] = pl_smplxbeta.detach().clone()
            
    # for v_id in tqdm(range(pl_verts0.shape[0])):
    #     image0 = cv2.imread(ds.get_j2d_only(v_id)["im0_path"])/255.
    #     image1 = cv2.imread(ds.get_j2d_only(v_id)["im1_path"])/255.
    #     vis0 = renderer0(pl_verts0[v_id],
    #                                 cam0_extr[:3,3].detach().clone().cpu(),
    #                                 cam0_extr[:3,:3].detach().clone().cpu(),
    #                                 image0)
    #     vis1 = renderer1(pl_verts1[v_id],
    #                                     cam1_extr[:3,3].detach().clone().cpu(),
    #                                     cam1_extr[:3,:3].detach().clone().cpu(),
    #                                     image1)
    #     # vis0 = kp_viz(vis0,joints2d0[0])
    #     # vis1 = kp_viz(vis1,joints2d1[0])
    #     cv2.imwrite("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/"+viz_dir+"/{:06d}".format(v_id)+".jpg",np.concatenate([vis0[::3,::3],vis1[::3,::3]],axis=0)*255)

    # np.savez("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/"+viz_dir+"_"+str(begin)+"_"+str(end)+".npz",
    #             thetas=pl_smplxtheta_3d.detach().cpu().numpy(),
    #             betas=pl_smplxbeta.detach().cpu().numpy(),
    #             root_rot0=pl_smplxphi0_9d.detach().cpu().numpy(),
    #             root_rot1=pl_smplxphi1_9d.detach().cpu().numpy(),
    #             trans0=pl_smplxtau0.detach().cpu().numpy(),
    #             trans1=pl_smplxtau1.detach().cpu().numpy())
        
    # %% Vizualization generation
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,2,1,projection="3d")
    ax.scatter(cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),
                cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),
                cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),
                c=range(np.sum(robust_idcs)),cmap="viridis")
    # ax.plot(cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),
    #             cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),
    #             cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy())
    # arr_ends = torch.matmul(cam1_wrt_cam0,torch.tensor([0,0,1,1]).float().to(device).unsqueeze(1)).detach().cpu().numpy()[:,:,0]
    ax.set_xlim(-15,12)
    ax.set_ylim(-12,10)
    ax.set_zlim(0,20)
    # ax.xaxis.set_label_text("x(m)",{"fontsize":"large","fontweight":"bold"})
    # ax.yaxis.set_label_text("y(m)",{"fontsize":"large","fontweight":"bold"})
    # ax.zaxis.set_label_text("z(m)",{"fontsize":"large","fontweight":"bold"})
    ax.xaxis.set_ticks(list(range(-17,10,5)))
    ax.yaxis.set_ticks(list(range(-12,10,5)))
    ax.zaxis.set_ticks(list(range(0,20,5)))
    ax.view_init(-28,-91)
    # arr_ends = transforms.rotation_conversions.matrix_to_axis_angle(cam1_wrt_cam0[:,:3,:3]).detach().cpu().numpy()
    # ax.quiver(cam1_wrt_cam0[::100,0,3].detach().cpu().numpy(),
    #             cam1_wrt_cam0[::100,1,3].detach().cpu().numpy(),
    #             cam1_wrt_cam0[::100,2,3].detach().cpu().numpy(),
    #             arr_ends[::100,0],arr_ends[::100,1],arr_ends[::100,2],length=9.0,normalize=True,arrow_length_ratio=0.15)
    ax = fig1.add_subplot(1,2,2,projection="3d")
    ax.scatter(pl_cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),
                pl_cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),
                pl_cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),
                c=range(np.sum(robust_idcs)),cmap="viridis")
    # ax.plot(pl_cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),
    #             pl_cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),
    #             pl_cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy())
    ax.set_xlim(-15,12)
    ax.set_ylim(-12,10)
    ax.set_zlim(0,20)
    ax.xaxis.set_ticks(list(range(-17,10,5)))
    ax.yaxis.set_ticks(list(range(-12,10,5)))
    ax.zaxis.set_ticks(list(range(0,20,5)))
    ax.view_init(-28,-91)
    # plt.show()
    # print('ax.azim {}'.format(ax.azim))
    # print('ax.elev {}'.format(ax.elev))

    fig2 = plt.figure()            
    ax = fig2.add_subplot(1,2,1)
    ax.plot(ds.apose[:,:6990][0,robust_idcs,:,2].sum(1))
    ax.plot(smpl_wrt_cam0[robust_idcs,:3,3].detach().cpu().numpy())
    ax = fig2.add_subplot(1,2,2)
    ax.plot(ds.apose[:,:6990][0,robust_idcs,:,2].sum(1))
    ax.plot(pl_smpl_wrt_cam0[robust_idcs,:3,3].detach().cpu().numpy())
    
    fig3 = plt.figure()
    ax = fig3.add_subplot(1,2,1)
    trunc_apose = ds.apose[:,:6990]
    plt.plot(trunc_apose[:,robust_idcs][:,:,:,2].sum(2).sum(0))
    plt.plot(cam1_wrt_cam0[robust_idcs,:3,3].detach().cpu().numpy())
    ax = fig3.add_subplot(1,2,2)
    plt.plot(trunc_apose[:,robust_idcs][:,:,:,2].sum(2).sum(0))
    plt.plot(pl_cam1_wrt_cam0[robust_idcs,:3,3].detach().cpu().numpy())

    fig5 = plt.figure()
    ax = fig5.add_subplot(3,1,1)
    plt.plot(bl_cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),".",markersize=1,mec="indianred",mfc="indianred")
    plt.plot(cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),".",markersize=1,mec="chartreuse",mfc="chartreuse")
    # plt.plot(pl_cam1_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),".",markersize=1,mec="blue",mfc="blue")
    ax.yaxis.set_label_text("x(m)",{"fontsize":"large","fontweight":"bold"})
    ax.legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
    # ax.set_xlim(0,7000)
    ax.xaxis.set_ticks([])
    ax = fig5.add_subplot(3,1,2)
    plt.plot(bl_cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),".",markersize=1,mec="seagreen",mfc="seagreen")
    plt.plot(cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),".",markersize=1,mec="fuchsia",mfc="fuchsia")
    # plt.plot(pl_cam1_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),".",markersize=1,mec="cyan",mfc="cyan")
    ax.yaxis.set_label_text("y(m)",{"fontsize":"large","fontweight":"bold"})
    ax.legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
    # ax.set_xlim(0,7000)
    ax.xaxis.set_ticks([])
    ax = fig5.add_subplot(3,1,3)
    plt.plot(bl_cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),".",markersize=1,mec="dodgerblue",mfc="dodgerblue")
    plt.plot(cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),".",markersize=1,mec="mediumvioletred",mfc="mediumvioletred")
    # plt.plot(pl_cam1_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),".",markersize=1,mec="orange",mfc="orange")
    ax.yaxis.set_label_text("z(m)",{"fontsize":"large","fontweight":"bold"})
    ax.xaxis.set_label_text("frame number",{"fontsize":"large","fontweight":"bold"})
    ax.legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
    # ax.set_xlim(0,7000)
    plt.subplots_adjust(wspace=0, hspace=0.02)

    fig4 = plt.figure()            
    ax = fig4.add_subplot(1,2,1)
    ax.plot(ds.apose[:,:6990][1,robust_idcs,:,2].sum(1))
    ax.plot(smpl_wrt_cam1[robust_idcs,:3,3].detach().cpu().numpy())
    ax = fig4.add_subplot(1,2,2)
    ax.plot(ds.apose[:,:6990][1,robust_idcs,:,2].sum(1))
    ax.plot(pl_smpl_wrt_cam1[robust_idcs,:3,3].detach().cpu().numpy())
    

    # %%
    # from scipy.signal import savgol_filter
    # pl_cam1_wrt_cam0[:,:3,3] = torch.from_numpy(savgol_filter(pl_cam1_wrt_cam0[:,:3,3].detach().cpu().numpy(),1001,2,axis=0)).float().to(device)
    # for begin in tqdm([0,2000,4000,6000]):

    #     if begin == 6000:
    #         end = 6990
    #     else:
    #         end = begin + 2000
    #     lseq = end-begin

    #     # %% all the vairables
    #     cam1_extr = torch.eye(4,device=device).float()
    #     smplxbeta = torch.zeros([10],device=device).float().clone()

    #     # fix parameters
    #     cam0_extr = torch.eye(4,device=device).float()

    #     sub_err_idcs = err_idcs[begin:end]
    #     sub_robust_idcs = robust_idcs[begin:end]
        

    #     # main loop
    #     with autograd.detect_anomaly():
    #         pl_smplxtheta = pl_smpl_z[begin:end].detach().clone()
    #         pl_smplxtheta.requires_grad = True
    #         pl_smplxphi0 = transforms.matrix_to_rotation_6d(pl_smpl_wrt_cam0[begin:end,:3,:3]).detach().clone()
    #         pl_smplxphi0.requires_grad = True
    #         pl_smplxtau0 = pl_smpl_wrt_cam0[begin:end,:3,3].detach().clone()
    #         pl_smplxtau0.requires_grad = True
    #         pl_cam1_extr_rot = transforms.matrix_to_rotation_6d(torch.inverse(pl_cam1_wrt_cam0[begin:end])[:,:3,:3]).detach().clone()
    #         pl_cam1_extr_rot.requires_grad = True
            
    #         pl_cam1_extr_trans = torch.inverse(pl_cam1_wrt_cam0[begin:end])[:,:3,3].detach().clone()
    #         n_iters = 300

    #         # create optimizer
    #         optim2 = torch.optim.Adam([pl_smplxtheta,
    #                                 pl_smplxphi0,
    #                                 pl_smplxtau0,
    #                                 pl_cam1_extr_rot,
    #                                 pl_smplxbeta],
    #                                 lr=0.01)
    #         optim = optim2

    #         for j in tqdm(range(n_iters)):
                
    #             pl_smplxtheta_3d = vp_model.decode(pl_smplxtheta)["pose_body"].reshape(-1,63)
                
    #             # forward SMPLX
    #             smplx_out = smplx_model.forward(betas=pl_smplxbeta.unsqueeze(0).expand([lseq,-1]), 
    #                                         pose_body=pl_smplxtheta_3d,
    #                                         root_orient=torch.zeros(lseq,3,device=device).float(),
    #                                         trans = torch.zeros(lseq,3,device=device).float())
    #             pl_smplxphi0_9d = transforms.rotation_6d_to_matrix(pl_smplxphi0).squeeze(0)
    #             pl_cam1_extr_rot_9d = transforms.rotation_6d_to_matrix(pl_cam1_extr_rot).squeeze(0)

    #             transf_mat0 = torch.cat([pl_smplxphi0_9d[:,:3,:3],
    #                                 pl_smplxtau0.unsqueeze(2)],dim=2)
    #             verts0,joints3d0,_,_ = transform_smpl(transf_mat0,
    #                                     smplx_out.v,
    #                                     smplx_out.Jtr)

    #             # joints3d0 = torch.matmul(j_regressor,verts0)
    #             # joints3d1 = torch.matmul(j_regressor,verts1)
    #             joints2d0 = perspective_projection(joints3d0[:,:24],
    #                                                     rotation=cam0_extr[:3,:3].unsqueeze(0).expand([lseq,-1,-1]),
    #                                                     translation=cam0_extr[:3,3].expand([lseq,-1]),
    #                                                     focal_length=[intr0[0,0],intr0[1,1]],
    #                                                     camera_center=intr0[:2,2]).squeeze(0)
                
    #             joints2d1 = perspective_projection(joints3d0[:,:24],
    #                                                     rotation=pl_cam1_extr_rot_9d,
    #                                                     translation=pl_cam1_extr_trans,
    #                                                     focal_length=[intr1[0,0],intr1[1,1]],
    #                                                     camera_center=intr1[:2,2]).squeeze(0)

    #             sigma = sigma2d
    #             loss_fun = torch.nn.MSELoss(reduction="none")
                
    #             joints2d_gt0[:,:,[1,2],2:] /= 2   # less weight for hips
    #             joints2d_gt1[:,:,[1,2],2:] /= 2
                
    #             # loss_2d = (joints2d_gt0[begin:end][sub_err_idcs,0,:,2:]*loss_fun(joints2d0[sub_err_idcs],joints2d_gt0[begin:end][sub_err_idcs,0,:,:2])).mean() + \
    #             #             (joints2d_gt1[begin:end][sub_err_idcs,0,:,2:]*loss_fun(joints2d1[sub_err_idcs],joints2d_gt1[begin:end][sub_err_idcs,0,:,:2])).mean() + \
    #             loss_2d =     (joints2d_gt0[begin:end][sub_robust_idcs,1,:,2:]*loss_fun(joints2d0[sub_robust_idcs],joints2d_gt0[begin:end][sub_robust_idcs,1,:,:2])).mean() + \
    #                         (joints2d_gt1[begin:end][sub_robust_idcs,1,:,2:]*loss_fun(joints2d1[sub_robust_idcs],joints2d_gt1[begin:end][sub_robust_idcs,1,:,:2])).mean()

                

    #             loss_vposer = torch.mul(pl_smplxtheta,pl_smplxtheta).mean()

    #             loss_beta = torch.mul(smplxbeta,smplxbeta).mean()
                
                
    #             sub_robust_idcs_temporal = np.logical_and(sub_robust_idcs[:-1],sub_robust_idcs[1:])
    #             loss_temporal = 10*loss_fun(pl_smplxtheta_3d[1:],pl_smplxtheta_3d[:-1])[sub_robust_idcs_temporal].mean() + \
    #                             100*loss_fun(pl_smplxphi0[1:],pl_smplxphi0[:-1])[sub_robust_idcs_temporal].mean() + \
    #                             100*loss_fun(pl_cam1_extr_rot[1:],pl_cam1_extr_rot[:-1])[sub_robust_idcs_temporal].mean() + \
    #                             100*loss_fun(pl_smplxtau0[1:],pl_smplxtau0[:-1])[sub_robust_idcs_temporal].mean()  

    #             loss = loss_2d + w_beta*loss_beta + w_vposer*loss_vposer + w_temporal*loss_temporal
                
    #             optim.zero_grad()
    #             loss.backward()
    #             optim.step()

    #     pl_smpl_z[begin:end,:] = pl_smplxtheta.detach().clone()
    #     pl_smpl_wrt_cam0[begin:end,:3,:3] = pl_smplxphi0_9d.detach().clone()
    #     pl_smpl_wrt_cam0[begin:end,:3,3] = pl_smplxtau0.detach().clone()
    #     import ipdb;ipdb.set_trace()
    #     pl_cam1_wrt_cam0[begin:end,:3,:3] = torch.inverse(pl_cam1_extr_rot_9d).detach().clone()
    #     pl_verts0[begin:end] = verts0.detach().cpu().numpy()


    # %% baseline and airpose results load
    images0 = np.array(ds.db["im0"])
    images1 = np.array(ds.db["im1"])

    airpose_pred_vertices_cam0 = torch.cat([i["output"]["pred_vertices_cam0"].to("cuda") for i in res[res_id]])
    airpose_pred_vertices_cam1 = torch.cat([i["output"]["pred_vertices_cam1"].to("cuda") for i in res[res_id]])

    fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/hmr/version_2_from_newlytrinedckpt/checkpoints/epoch=388.pkl"
    fname0 = fname + "0"
    fname1 = fname + "1"
    res0 = pkl.load(open(fname0,"rb"))
    res1 = pkl.load(open(fname1,"rb"))
    bl_pred_vertices_cam0 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res0[res_id]])
    bl_pred_vertices_cam1 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res1[res_id]])

    samples = [200,5300,800,4700,6700]
    ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
    ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)

    bl_rend_ims0 = renderer0.visualize_tb(bl_pred_vertices_cam0[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0)
    bl_rend_ims1 = renderer1.visualize_tb(bl_pred_vertices_cam1[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims1)
    airpose_rend_ims0 = renderer0.visualize_tb(airpose_pred_vertices_cam0[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0)
    airpose_rend_ims1 = renderer1.visualize_tb(airpose_pred_vertices_cam1[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims1)