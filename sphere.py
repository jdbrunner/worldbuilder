import sys
import numpy as np

from matplotlib import cm
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy import geometry

canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                           size=(800, 600), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'
# view.camera = 'fly'
#

Globe = np.load(sys.argv[1])
Elevation = np.load(sys.argv[2])
Temps = np.load(sys.argv[3])

if len(sys.argv) > 4:
    Rads =np.ones(Globe[0].shape)+0.01*Elevation
else:
    Rads = np.ones(Globe[0].shape)

colors = np.empty((*Elevation.shape,4))
colors[Elevation>0.45] = cm.YlGn_r(Elevation)[Elevation>0.45]
colors[Elevation<=0.45] = cm.Blues_r(Elevation)[Elevation<=0.45]

Tempcolors = np.zeros_like(colors)
Tempcolors[np.where(Temps < 0.15)] = cm.cool(Temps, alpha = 0.8)[np.where(Temps < 0.15)]

print(Tempcolors)

def toCart(p,t,r):
    return r*np.cos(t)*np.sin(p),r*np.sin(t)*np.sin(p),r*np.cos(p)

BumpySphereCart = toCart(*Globe,Rads)
BumpySphereCart2 = toCart(*Globe,Rads+0.001)

# bumpy = geometry.MeshData(vertices = BumpySphereCart)
scene.visuals.GridMesh(*BumpySphereCart,parent=view.scene, colors = colors,shading = None)
scene.visuals.GridMesh(*BumpySphereCart2,parent=view.scene, colors = Tempcolors,shading = None)


view.camera.set_range(x=[-1.1, 1.1])

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()
