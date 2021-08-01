import numpy as np
from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface #import before pyrender
import pyrender
import skimage
import skimage.measure
import trimesh

from tensor_decomp import apply_tt, orthogonalize, apply_parafac

# Constants
ORIGIN = np.array((0, 0, 0))
X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))
UP = Z
DEFAULT_BOUNDARIES = \
    np.asarray(
                [(-5, 5),   # x
                (-5, 5),   # y
                (-5, 5)]   # z
            ) # 3x2, columns: (min, max) in world coordinates in meters

# Helper functions 
def _length(a):
    return np.linalg.norm(a, axis=1)

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

# def scale(other, factor):
#     try:
#         x, y, z = factor
#     except TypeError:
#         x = y = z = factor
#     s = (x, y, z)
#     m = min(x, min(y, z))
#     def f(p):
#         return other(p / s) * m
#     return f

def generate_sdf_rep(obstacles, grid_bounds=None, voxel_res=0.2, tt_rank=2, union_approx=True, 
                     use_cp=False, x0=None, x_target=None, sdf_only=False): 
    """
    Function to generate sdf representation of environment. 
    args: 
        voxel_res: voxel resolution in grid 
        tt_rank: rank for tensor train decomposition -> should be as small as possible
    """
    if grid_bounds is None: 
        grid_bounds =  \
            np.asarray(
                        [(min(x0[0], x_target[0]), max(x0[0], x_target[0])),   # x
                        (min(x0[1], x_target[1]), max(x0[1], x_target[1])),   # y
                        (min(x0[2], x_target[2]), max(x0[2], x_target[2]))]   # z
                    ) # 3x2, columns: (min, max) in world coordinates in meters
    # init environment building class 
    env_builder = EnvCreator(vol_bnds=grid_bounds, voxel_size=voxel_res)
    
    sdf_vol = None
    for i, obs in enumerate(obstacles): 
        params = {'radius':obs['rad'], 'center':obs['pos']}
        
        # generate grid-based sdf representation for given obstacle 
        sdf_vol_temp = sdf_vol # keep copy of previous volume to perform union operation 
        sdf_vol = env_builder.generate_sdf_vol_parametric(primitive='sphere', param_dct=params)
        
        if sdf_vol_temp is not None: 
            sdf_vol = env_builder.union([sdf_vol_temp, sdf_vol], approx=union_approx)
        
    # compress sdf-volume by tensor train factorization 
    if not use_cp and not sdf_only: 
        tt_sdf_vol = apply_tt(sdf_vol, rank=tt_rank) #.copy(order='C').astype(np.float32)
        tt_sdf_orth = orthogonalize(tt_sdf_vol)
        return sdf_vol, tt_sdf_vol, tt_sdf_orth, env_builder
    elif use_cp and not sdf_only: 
        cp_sdf_vol = apply_parafac(sdf_vol, rank=tt_rank)
        return sdf_vol, cp_sdf_vol, env_builder
    else: # return sdf only 
        return sdf_vol, env_builder

class EnvCreator(object): 
    def __init__(self, vol_bnds=DEFAULT_BOUNDARIES, voxel_size=0.2, analytic=True):
        if analytic:  
            self.voxel_size = voxel_size
            self.vol_bnds = np.asarray(vol_bnds)
            self.vol_dim = np.ceil((self.vol_bnds[:,1] - self.vol_bnds[:,0]) / self.voxel_size).copy(order='C').astype(int) # array with dimensions of voxel grid 
            self.voxel_grid = self.generate_voxel_grid()
        else: # create environment from mesh files 
            self.voxel_res = 64


    def sdf3_sphere(self, p, radius, center, rot_angle=None, rot_axis=None,  t=None, verbose=True):
        if verbose: 
            print("Input: Voxel grid coordinates with shape {}".format(p.shape))
        if rot_angle and rot_axis: 
            p = self.rotate_points(p, rot_angle, rot_axis)
        if t: # TODO: what if rotation and translation given -> apply them in which order? 
            p = self.translate_points(p, t)
        val = _length(p - center) - radius
        val = val.reshape((-1, 1))
        if verbose:
            print("Output: SDF Voxels with shape {}".format(val.shape))
        return val

    def sdf3_ellipsoid(self, p, size, rot_angle=None, rot_axis=None,  t=None):
        """ Returns 3D signed distance function for an ellipsoid given a 3D a point set (centered aroun origin).
            NOTE: bound sdf (lower bound to real SDF, since it can only be approximated for ellipsoid primitive).
        Args:
            p (numpy ndarray): 3D points (e.g. center points of voxel grid)
            size (tuple): radius in x,y,z direction (1,1,1) will return a unit sphere. 

        Returns:
            SDF values for given point set (nx1) with n being the number of points p 
        """
        print("Input: Voxel grid coordinates with shape {}".format(p.shape))
        if rot_angle and rot_axis: 
            p = self.rotate_points(p, rot_angle, rot_axis)
        if t: # TODO: what if rotation and translation given -> apply them in which order? 
            p = self.translate_points(p, t)
        size = np.array(size)
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        val =  k0 * (k0 - 1) / k1
        val = val.reshape((-1, 1))
        print("Output: SDF Voxels with shape {}".format(val.shape))
        return val

    def translate_points(self, p, offset):
        return p - offset

    def rotate_points(self, p, angle, axis=Z):
        x, y, z = _normalize(axis)
        s = np.sin(angle)
        c = np.cos(angle)
        m = 1 - c
        matrix = np.array([
            [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
            [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
            [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
        ]).T
        return np.dot(p, matrix)

    def union(self, sdf_vols, approx=True): 
        base_vol = sdf_vols[0]
        for vol in sdf_vols[1:]: 
            if approx: # causing wrong distances inside objects (negative signs)
                base_vol = np.minimum(base_vol, vol)
            else: # exact case by case implementation 
                # TODO: implement properly - this is a draft, not debugged yet 
                # inflate volumes by another dimension to compare values 
                compare_vol = np.concatenate((base_vol.reshape(*base_vol.shape, 1), vol.reshape(*vol.shape, 1)), axis=3)
                # base_vol_idx = np.argmin(np.absolute(compare_vol), axis=3)
                compare_vol_idx = np.expand_dims(np.argmin(np.absolute(compare_vol), axis=3), axis=3)
                # base_vol = compare_vol[:,:, :, compare_vol_idx].reshape(*base_vol.shape)
                base_vol = np.take_along_axis(compare_vol, compare_vol_idx, axis=3).reshape(*base_vol.shape)

        return base_vol 

    def load_mesh(self, obj_type): 
        mesh = trimesh.load('objects/{}.obj'.format(obj_type), force='mesh')
        if mesh.is_watertight: 
            print('[OK] mesh is watertight')
        else: 
            print('[WARNING] mesh is not watertight and cause artifacts when rendering to sdf volume')
        return mesh

    def merge_sdf_volumes(self, vol1, vol2):
        # build cartesian product of vol1 and vol2 
        
        # perform elementwise min on absolut sdf values 
        keep_idx = np.argmin() 
        pass 

    def generate_sdf_view_from_mesh(self, meshes, positions): 
        scene = pyrender.Scene()
        for mesh, pos in zip(meshes, positions): 
            points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
            colors = np.zeros(points.shape)
            colors[sdf < 0, 2] = 1  # blue where the SDF = negative (inside object) 
            colors[sdf > 0, 0] = 1  # red where the SDF = positive  (outside object)
            colors[sdf == 0,1] = 1  # green on boundary 
            cloud = pyrender.Mesh.from_points(points, colors=colors)
            scene.add(cloud, pose=pos)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=100, viewer_flags={'view_center':(0,0,0)})

    def generate_voxel_vol_from_mesh(self, mesh): 
        voxels = mesh_to_voxels(mesh, self.voxel_res, pad=True)
        return voxels

    def render_voxel_view(self, voxel_vol, return_mesh=False, return_points=False):
        vertices, faces, normals, _ = skimage.measure.marching_cubes(voxel_vol, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        if not return_mesh and not return_points: 
            mesh.show()
        elif return_points and not return_mesh:
            return vertices[faces].reshape((-1, 3))
        else: 
            return mesh 

    def generate_voxel_grid(self): 
        # build voxelgrid in world coordinates, i.e. world coordinates 
        assert self.vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Adjust volume bounds and ensure C-order contiguous
        self.vol_bnds[:,1] = self.vol_bnds[:,0] + self.vol_dim * self.voxel_size
        vol_origin = self.vol_bnds[:,0].copy(order='C').astype(np.float32)
        
        # voxel grid 
        x_ = np.linspace(*self.vol_bnds[0], int(self.vol_dim[0]))
        y_ = np.linspace(*self.vol_bnds[1], int(self.vol_dim[1]))
        z_ = np.linspace(*self.vol_bnds[2], int(self.vol_dim[2]))
        i, j, k = np.meshgrid(x_, y_, z_, indexing='ij')

        assert np.all(i[:,0,0] == x_)
        assert np.all(j[0,:,0] == y_)
        assert np.all(k[0,0,:] == z_)

        voxel_grid = np.vstack([i.ravel(), j.ravel(), k.ravel()]).T

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
        range(self.vol_dim[0]),
        range(self.vol_dim[1]),
        range(self.vol_dim[2]),
        indexing='ij'
        )

        vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
        ], axis=0).astype(int)

        return voxel_grid

    def generate_sdf_vol_parametric(self, primitive='sphere', param_dct=None, truncated=False, verbose=False):
        if truncated: 
            # TODO: implement truncated SDF (memory efficiency)
            # define truncation margin 
            trunc_margin = 10 * self.voxel_size  # truncation on SDF 
        if primitive == 'sphere': 
            # create sdf_volume 
            if not param_dct: 
                radius=1
                center=ORIGIN
            else: 
                radius = param_dct['radius']
                center = param_dct['center']
            sdf_vol = self.sdf3_sphere(self.voxel_grid, radius, center, verbose=verbose).reshape(self.vol_dim)
        elif primitive == 'ellipsoid': 
            if not param_dct: 
                size = (1,1,1)
            else: 
                size = param_dct['size']
            sdf_vol = self.sdf3_ellipsoid(self.voxel_grid, size).reshape(self.vol_dim)
        else: 
            raise NotImplementedError
        return sdf_vol

    def compare_mesh_view(self, meshes, translations, colors, return_scene=False): 
        scene = pyrender.Scene()

        scene = trimesh.Scene()
        for mesh, translation, color in zip(meshes, translations, colors):     
            mesh.visual.face_colors = color
            scene.add_geometry(mesh)
            scene.apply_translation(translation)

        # axis = trimesh.creation.axis(origin_color=[1., 0, 0])
        # scene.add_geometry(axis)
        if return_scene: 
            return scene
        else: 
            scene.show()


def create_envs_from_files():
    env_builder = EnvCreator(analytic=False)
    
    objects = ['chair', 'shelf', 'table'] # 'bottle'
    
    positions = [
        np.eye(4), 
        [   [1., 0., 0., 2.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], 
        [   [1., 0., 0., -2.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]
    ]

    meshes = []

    for obj in objects: 
        mesh = env_builder.load_mesh(obj)
        meshes.append(mesh)
        env_builder.generate_voxel_vol_from_mesh(mesh)  
    env_builder.generate_sdf_view_from_mesh(meshes, positions)


def create_analytical_env(): 
    env_builder = EnvCreator()
    translations = [
        [-10, 0, 0], 
        [10, 0, 0]
    ]
    colors = [
        [0, 1., 0, 0.5], # green original
        [0, 0, 1., 0.5] # blue compressed 

    ]
    
    voxel_vol = env_builder.generate_sdf_vol_parametric(primitive='ellipsoid')
    mesh = env_builder.render_voxel_view(voxel_vol, return_mesh=True)

    voxel_vol_compressed = apply_tt(voxel_vol, rank=4).copy(order='C').astype(np.float32)
    mesh_compressed = env_builder.render_voxel_view(voxel_vol_compressed, return_mesh=True)

    print(voxel_vol_compressed.shape)
    env_builder.compare_mesh_view([mesh, mesh_compressed], translations, colors)

    scene = trimesh.Scene()

    env_builder.render_voxel_view(voxel_vol)
    env_builder.render_voxel_view(voxel_vol_compressed)


if __name__ == "__main__":
    create_analytical_env()
