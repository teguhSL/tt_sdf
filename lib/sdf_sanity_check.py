import os
import sys 

import numpy as np
import trimesh
import skimage

from mesh_to_sdf import mesh_to_voxels

from env_creator import EnvCreator, generate_sdf_rep
from tensor_decomp import apply_tt, uncompress_tt_rep
from visualization_utils import plot_traj_projections, plot_traj_and_obs_3d, show_single_layer
from mesh import write_binary_stl



DEFAULT_BOUNDS = np.asarray([(-2,2),   # x
                             (-2,2),   # y
                             (-2,2)]   # z
            ) # 3x2, columns: (min, max) in world coordinates in meters
save_path = os.path.join(os.getcwd(), 'data/sdf_test_meshes')


def create_test_env(obstacles, x0, xT, voxel_res=0.05, tt_rank=2, save_path=None): 
    sdf_vol, tt_factors, _, env_builder = generate_sdf_rep(obstacles, grid_bounds=DEFAULT_BOUNDS, 
                                                        voxel_res=voxel_res, tt_rank=tt_rank, 
                                                        union_approx=True)    

    show_single_layer(sdf_vol, int(sdf_vol.shape[2]/2), save_path=save_path)
    return sdf_vol, tt_factors, env_builder
    

def create_analytic_test_env(rank, exp_name): 
    os.makedirs(save_path, exist_ok=True)

    xT = np.array([1, 1, 1, 0,0,0])
    x0 = np.array([-1,-1,-1,0,0,0])

    # obstacles = [
    #     {'pos': [0.8, 0.8, 0.8], 'rad': 0.3}, 
    #     {'pos': [0., 0.4, 0.], 'rad': 0.1}, 
    #     {'pos': [0.4, 0., 0.], 'rad': 0.2}, 
    #     {'pos': [0.2, 0.5, 0.5], 'rad': 0.3}
    # ]
    obstacles = [
        {'pos': [-0.2, -0.3, -0.3], 'rad': 0.7}, 
        # {'pos': [ 0.3, 0.1, -0.1], 'rad': 0.5}, 
        # {'pos': [ 0.2, -0.1, 0.2], 'rad': 0.6}, 
        {'pos': [ 0.6, 0.5, 0.4], 'rad': 0.4}
    ]

    fig_path = os.path.join(save_path, 'sdf_%s.pdf'%exp_name)
    sdf_vol, tt_factors, env_builder = create_test_env(obstacles, x0, xT, tt_rank=rank, save_path=fig_path)
    reconstructed_sdf = uncompress_tt_rep(tt_factors, F=sdf_vol)

    sdf_mesh = env_builder.render_voxel_view(sdf_vol, return_mesh=False, return_points=True)
    sdf_mesh_reconstr = env_builder.render_voxel_view(reconstructed_sdf, return_mesh=False, return_points=True)

    write_binary_stl(os.path.join(save_path, '%s.stl'%exp_name), sdf_mesh)
    write_binary_stl(os.path.join(save_path, '%s_reconstr.stl'%exp_name), sdf_mesh)


def sdf_from_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    if mesh.is_watertight: 
        print('[OK] mesh is watertight')
    else: 
        print('[WARNING] mesh is not watertight and cause artifacts when rendering to sdf volume')

    sdf = mesh_to_voxels(mesh, voxel_resolution=256, pad=True)#, check_result=True)
    return sdf


def test_sdf_from_mesh(tt_rank, mesh_path, exp_name='mesh_sdf'): 
    sdf = sdf_from_mesh(mesh_path)
    tt_factors = apply_tt(sdf, rank=tt_rank, verbose=True)
    sdf_reconst = uncompress_tt_rep(tt_factors, F=sdf)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf_reconst, level=0)
    sdf_reconst_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    points = vertices[faces].reshape((-1, 3))
    write_binary_stl(os.path.join(save_path, '%s_reconstr.stl'%exp_name), points)
    write_binary_stl(os.path.join(save_path, '%s_reconstr.stl'%exp_name), points)


if __name__=='__main__': 
    create_analytic_test_env(3, 'example_env4')
    # mesh_path = os.path.join(save_path, 'test_obj.stl')
    # # mesh_path = os.path.join(save_path, 'bookcase.stl')

    # # for r in range(20,30): 
    # r=3
    # print(r)

    # sdf = sdf_from_mesh(mesh_path)
    # vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
    # points = vertices[faces].reshape((-1, 3))
    # write_binary_stl(os.path.join(save_path, 'mesh_sdf_original.stl'), points)

    # test_sdf_from_mesh(r, mesh_path, 'mesh_sdf_voxelres512_r%d_1'%r)
    # test_sdf_from_mesh(r, mesh_path, 'shelf_mesh_sdf_r%d'%r)


