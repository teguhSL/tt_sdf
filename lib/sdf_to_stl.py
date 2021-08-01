import os
from sdf import * 

"""
Sanity check for mutiple obstacle sdf representations using the following library: 
https://github.com/fogleman/sdf
"""

ORIGIN = np.array((0, 0, 0))
path = os.path.join(os.getcwd(), 'data')

def test(): 
    a = sphere(radius=1, center=ORIGIN)
    b = sphere(radius=0.5, center=np.array((1,1,1)))

    

    f = union(a, b)
    f.save(os.path.join(path, 'sdf_test1.stl'), step=0.01)
    return f


if __name__ == '__main__': 
    f = test()
