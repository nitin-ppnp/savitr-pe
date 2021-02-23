"""
Geometry primitives and algorithms
"""

import numpy as np

def project_3d_points(intrinsic, extrinsic, points):
    '''
    Takes 3d points and intrinsic and extrinsic parameters
    of a camera. Gives the projected 2D point on the image plane.

    :param intrinsic: intrinsic parameters of the camera (3X3)
    :param extrinsic: extrinsic parameters of the camera (4X4)
    :param points: 3D points (3XN)

    '''

    x = np.zeros([2, points.shape[1]])

    intr = np.zeros([3, 4])
    intr[:, :3] = intrinsic

    for i in np.arange(points.shape[1]):

        point = np.append(points[:, i], 1)

        point_2d = np.matmul(np.matmul(intr, extrinsic), point)

        x[:, i] = point_2d[0:2] / point_2d[2]

    return x


def error_in_3d(j_2d, extrinsic, j_3d):
    '''
    ''' 

    ext = np.linalg.inv(extrinsic)

    # camera center in global coordinates
    pos = ext[:3,3]

    # 2D joint location in global coordinates
    pp = np.matmul(ext, np.append(j_2d,1))
    pp = pp[:3]/pp[3]
    
    # direction vector of two lines and perpendicular line
    v = pp - pos
    v = v/np.linalg.norm(v)
    # v3 = np.cross(v1,v2)
     
    # projection of j_3d on line
    j3d_p = pos + np.dot((j_3d - pos),v)*v

    # # closest points on both the lines
    # t = np.matmul(np.linalg.inv(np.column_stack([v1,-v2,v3])),(pos_m2 - pos_m1))
    # j3d_est_p1 = pos_m1 + t[0]*v1
    # j3d_est_p2 = pos_m2 + t[1]*v2
    # j3d_est = (j3d_est_p1 + j3d_est_p2)/2

    err = np.linalg.norm(j_3d-j3d_p)
    # err_est = np.linalg.norm(j3d_est-j3d_est_p1) + np.linalg.norm(j3d_est-j3d_est_p2)

    # import ipdb; ipdb.set_trace()

    return err 

def lstsq_triangulation(intrinsic,extrinsic,points_2d,probs):
    '''
    '''
    
    cams = points_2d.shape[0]
    NNs = points_2d.shape[1]
    
    error = np.zeros([cams,NNs])

    a=[]
    b=[]
    norm_points = []
    w = []

    for cam in range(cams):

        for nn in range(NNs):

            extr = extrinsic[cam,nn,:3,:]
            # convert to homogeneous coordinates
            point = np.append(points_2d[cam,nn,:], 1)

            # normalize the 2D points using the intrinsic parameters
            norm_points.append(np.matmul(np.linalg.inv(intrinsic[cam,nn,:,:]), point))

            # we'll use equation 14.42 of the CV book by Simon Prince.
            # form of the equation is Ax=b.

            # generate the matrix A and vector b
            a.append(np.outer((norm_points[-1])[0:2], extr[2, 0:-1]) - extr[0:2, 0:-1])
            b.append(extr[0:-1, -1] - extr[-1, -1] * (norm_points[-1])[0:2])
            w.append([probs[cam,nn],probs[cam,nn]])
    
    W = np.sqrt(np.concatenate(w))
    A = np.concatenate(a)
    B = np.concatenate(b)
    
    Aw = A * W[:,np.newaxis]
    Bw = B * W

    # solve with least square estimate
    x = np.linalg.lstsq(Aw, Bw,rcond=None)[0]

    return x, norm_points


def get_single_3d_point(intrinsic,extrinsic,points_2d,probs):
    '''
    '''
    
    x, norm_points = lstsq_triangulation(intrinsic,extrinsic,points_2d,probs)

    for cam in range(cams):
        for nn in range(NNs):
            # error in 3d
            error[cam,nn] = error_in_3d(norm_points[cam*NNs+nn],extrinsic[cam,nn,:,:] ,x)
    
    pl = error > 0.4
    if pl.any():
        probs[pl] = 0
        x, error = get_single_3d_point(intrinsic,extrinsic,points_2d,probs)

    return x, error


def get_3d_points(intrinsic, extrinsic, points_2d):
    '''
    Takes 2d points and intrinsic and extrinsic parameters
    of two cameras. Gives the least square estimate of the points in 3D.

    :param intrinsic1: intrinsic parameters camera1 (CXNNX3X3)
    :param extrinsic1: inverse of extrinsic parameters camera1 (CXNNX4X4)
    :param points1: 2D points in the image plane of camera1 (CXNNXNX2)
    :returns: point in 3D world coordinate
    '''

    cams = points_2d.shape[0]
    NNs = points_2d.shape[1]
    n_joints = points_2d.shape[2]

    x = np.zeros([n_joints,3])
    error = np.zeros([cams,NNs,n_joints])

    for i in np.arange(n_joints):

        x[i,:], error[:,:,i] = get_single_3d_point(intrinsic,extrinsic,points_2d[:,:,i,:],probs[:,:,i])


    return (x, error)

def center_of_mass(joint3d):
    '''
    return center of mass of given set of 3D points
    '''

    return np.mean(joint3d,axis=1)
