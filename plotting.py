from matplotlib import cm
from vispy import scene
from geofuns import *
import numpy as np
import matplotlib.pyplot as plt

def mapThetas(Globe,bend = 0.5):
    return ((1-bend)+bend*np.sin(Globe[0]))*Globe[1]


def blat(ph,th):
    return np.array([np.sin(th),-np.cos(th),0])
def blong(ph,th):
    return np.array([-np.cos(th)*np.cos(ph),-np.sin(th)*np.cos(ph),1])

def visWind(world,grid,bend = 0.5):
    windph = np.empty(grid[0].shape)
    windth = np.empty(grid[1].shape)
    AllWind = world.Wind(*grid)
    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            ph = grid[0][i][j]
            th = grid[1][i][j]
            latb = blat(ph,th)
            longb = blong(ph,th)
            wnd = np.array([AllWind[0][i][j],AllWind[1][i][j],AllWind[2][i][j]])
            combs = np.linalg.lstsq(np.array([latb,longb]).T,wnd,rcond = None)[0]
            windph[i][j] = combs[1]
            windth[i][j] = combs[0] + combs[1]*(1-bend)+bend*np.sin(ph)
    return windph,windth

def ColorWorld(world,cm1,cm2,rivers = False):

    Cols = np.empty(cm1(world.GlobeGrid[0]).shape)
    Cols[world.LandIndicator] = cm1(world.Elevation)[world.LandIndicator]
    Cols[np.invert(world.LandIndicator)] =cm2(world.Elevation)[np.invert(world.LandIndicator)]

    if rivers:
        RiverMsk = np.zeros(world.GlobeGrid[0].shape)
        for r in world.RiverIndices:
            RiverMsk[r] = 1
        RiverMsk = RiverMsk.astype(bool)

        Cols[RiverMsk] = np.array([[cm.Blues(0.9)]*world.GlobeGrid[0].shape[1]]*world.GlobeGrid[0].shape[0])[RiverMsk]

    return Cols

def RotateView(world,angle):

    numTheta = world.GlobeGrid[1].shape[0]

    MapTheta = mapThetas(world.GlobeGrid)

    RiverMsk = np.zeros(world.GlobeGrid[0].shape)
    for r in world.RiverIndices:
        RiverMsk[r] = 1
    RiverMsk = RiverMsk.astype(bool)

    Elev = world.Elevation.copy()
    LandInd = world.LandIndicator.copy()

    Elev = np.concatenate([Elev,Elev[1:]])
    LandInd = np.concatenate([LandInd,LandInd[1:]])
    RiverMsk = np.concatenate([RiverMsk,RiverMsk[1:]])

    angle = angle % 2*np.pi
    startAngle = angle - np.pi
    startRow = np.where(world.GlobeGrid[1] >= startAngle)[0][0]

    Elev = Elev[startRow:startRow + numTheta]
    LandInd = LandInd[startRow:startRow + numTheta]
    RiverMsk = RiverMsk[startRow:startRow + numTheta]

    fig = plt.figure(figsize = (10,5))
    ax = plt.axes()
    ax.set_ylim(np.pi,0)
    ax.scatter(MapTheta[LandInd],world.GlobeGrid[0][LandInd],s=20,c = Elev[LandInd], cmap = 'YlGn_r')
    ax.scatter(MapTheta[np.invert(LandInd)],world.GlobeGrid[0][np.invert(LandInd)],s=20,c = Elev[np.invert(LandInd)], cmap = 'Blues_r')
    ax.scatter(MapTheta[RiverMsk],world.GlobeGrid[0][RiverMsk],s=5,color = 'b')

    return None
