import numpy as np
import sys

from geofuns import *
from FantasyWorld import *
from plotting import *



World1 = FantasyWorld(GridSize = 200,oLevel = 0.3,riverProb = 0.02)





canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                           size=(800, 600), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'
# view.camera = 'fly'
#
avgTemp = (World1.Temps[0] + World1.Temps[1])/2
Cols2 = cm.coolwarm(avgTemp, alpha = 0.5)



Cols = np.empty(cm.YlGn_r(World1.GlobeGrid[0]).shape)
Cols[World1.LandIndicator] = cm.YlGn_r(World1.Elevation)[World1.LandIndicator]
Cols[np.invert(World1.LandIndicator)] = cm.Blues_r(World1.Elevation)[np.invert(World1.LandIndicator)]


RiverMsk = np.zeros(World1.GlobeGrid[0].shape)
for r in World1.RiverIndices:
    RiverMsk[r] = 1
RiverMsk = RiverMsk.astype(bool)

Cols[RiverMsk] = np.array([[cm.Blues(0.9)]*World1.GlobeGrid[0].shape[1]]*World1.GlobeGrid[0].shape[0])[RiverMsk]

Rads = np.ones(World1.GlobeGrid[0].shape) + 0.01*World1.Elevation
Rads2 = 1.01*np.ones(World1.GlobeGrid[0].shape)

BumpySphereCart = toCart(*World1.GlobeGrid,Rads + 0.01*World1.Elevation)
tempsphere = toCart(*World1.GlobeGrid,Rads2)
scene.visuals.GridMesh(*BumpySphereCart,parent=view.scene, colors = Cols,shading = None)
# scene.visuals.GridMesh(*tempsphere,parent=view.scene, colors = Cols2,shading = None)


view.camera.set_range(x=[-1.1, 1.1])

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()
