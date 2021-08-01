import numpy as np 
import meshio
from skimage import measure
import struct


def _mesh(points):
    points, cells = np.unique(points, axis=0, return_inverse=True)
    cells = [('triangle', cells.reshape((-1, 3)))]
    return meshio.Mesh(points, cells)


def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts[faces].reshape((-1, 3))


def save(points, path, *args, **kwargs):
    if path.lower().endswith('.stl'):
        write_binary_stl(path, points)
    else:
        mesh = _mesh(points)
        mesh.write(path)


def write_binary_stl(path, points):
    n = len(points) // 3

    points = np.array(points, dtype='float32').reshape((-1, 3, 3))
    normals = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
    normals /= np.linalg.norm(normals, axis=1).reshape((-1, 1))

    dtype = np.dtype([
        ('normal', ('<f', 3)),
        ('points', ('<f', (3, 3))),
        ('attr', '<H'),
    ])

    a = np.zeros(n, dtype=dtype)
    a['points'] = points
    a['normal'] = normals

    with open(path, 'wb') as fp:
        fp.write(b'\x00' * 80)
        fp.write(struct.pack('<I', n))
        fp.write(a.tobytes())

# volume = sdf(P).reshape((len(X), len(Y), len(Z)))
# points = _marching_cubes(volume)