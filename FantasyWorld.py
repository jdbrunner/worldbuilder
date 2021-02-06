#Defines the FantasyWorld class
from geofuns import *
from climatefuns import *
# from demographics import *
from race import *

import numpy as np

def GetNeighbors(l,size):
    if np.array(np.unravel_index(l,size))[1] == 0:
        return [np.ravel_multi_index((j,1),size) for j in range(size[1])]
    elif np.array(np.unravel_index(l,size))[1] == size[1]-1:
        return [np.ravel_multi_index((j,size[1]-1),size) for j in range(size[1])]
    else:
        return [np.ravel_multi_index(np.remainder(np.array(np.unravel_index(l,(100,100))) + [i,j],[size[0],size[1]+10]),size) for i in [-1,0,1] for j in [-1,0,1]]


class FantasyWorld:
    '''
    FantasyWorld - A procedurely generated world geography, climate, biomes, resources and demographic history.

    '''

    def __init__(self,GridSize = 100,radius = 3963, oLevel = 0.5,NumPlates = 50,dx = 0.01,minlen=0.5,maxlen=2,wigglyness = 1,smth = 0.2,jagged = 0.1,NumMts = 50,mtHeight = 0.5,mtdx = 0.01,mtminlen=0.1,mtmaxlen=2,mtwigglyness = 3,elevTemp = 0.3,winterVar = 3, summerVar = 2, extremes = 0.05, coastdecay = 8,coasteffect = 1, elevRain = 0.2, raincoastDecay = 4,raincoast = 0.4,riverProb = 0.1,R0scale = 20,N0scale = 20,Rrrate = 1):
        phi = np.linspace(0,np.pi,GridSize)
        theta = np.linspace(-np.pi,np.pi,GridSize)
        self.Radius = radius
        self.GlobeGrid = np.meshgrid(phi,theta)
        self.GridSize = (GridSize,GridSize)

        self.gridindices =  [(i,j) for i in range(self.GlobeGrid[0].shape[0]) for j in range(self.GlobeGrid[0].shape[1])]


        self.oLevel = oLevel
        self.ContinentalElevation, self.ContinentIndicator = MakeContinents(self,NumPlates = NumPlates,dx = dx,minlen=minlen,maxlen=maxlen,wigglyness = wigglyness,smth = smth,jagged = jagged)
        self.Elevation = self.ContinentalElevation
        self.Elevation,self.LandIndicator = AddMts(self,NumMts = NumMts,mtHeight = mtHeight,dx = mtdx,minlen=mtminlen,maxlen=mtmaxlen,wigglyness = mtwigglyness)
        self.Wind = SimpleWind
        self.oceans = np.array(list(zip(self.GlobeGrid[0][np.where(np.invert(self.LandIndicator))],self.GlobeGrid[1][np.where(np.invert(self.LandIndicator))])))
        self.Temps = baseTemp(self, m = elevTemp, winterVar = winterVar, summerVar = summerVar, extremes = extremes, coastdecay = coastdecay,coasteffect = coasteffect)
        self.RainFall = rainFall(self, m = elevRain, coastdecay = raincoastDecay,coasteffect = raincoast)
        self.RiverIndices,self.RiverLocs = makeRivers(self, riverProb = riverProb)
        self.RiverIndicator = np.array([a in self.RiverIndices for a in self.gridindices])
        self.OceanIndicator = np.invert(self.LandIndicator.flatten())

        self.GCDistance = GCdist

        self.NeighborDistances = np.array([[self.Radius*getDists((i,j),GridSize,GridSize,GCdist,self.GlobeGrid) for j in range(GridSize)] for i in range(GridSize)])

        coa =  np.array([[getSurrounding((i,j),self.LandIndicator) for j in range(self.GlobeGrid[0].shape[1])] for i in range(self.GlobeGrid[0].shape[0])])
        coas = coa*self.LandIndicator[:,:,None,None]
        self.CoastIndicator = np.array([[np.invert(coas[i,j]).any() for j in range(coas.shape[1])] for i in range(coas.shape[0])])

        InitialRenew = np.array([renew0(*p,self) for p in self.gridindices])
        self.InitialRenew = (R0scale*(InitialRenew - np.min(InitialRenew))/(np.max(InitialRenew)-np.min(InitialRenew))).reshape(GridSize,GridSize)
        self.Renewables = self.InitialRenew.copy()

        InitialNonRenew =np.array([initNonR(*p,self) for p in self.gridindices])
        self.InitialNonRenew =(N0scale*(InitialNonRenew-np.min(InitialNonRenew))/(np.max(InitialNonRenew)-np.min(InitialNonRenew))).reshape(GridSize,GridSize)
        self.NonRenewables = self.InitialNonRenew.copy()

        self.renewrate = Rrrate*np.ones(len(self.InitialRenew.flatten()))
        self.renewrate[self.OceanIndicator]  = 0
        self.renewrate = self.renewrate.reshape(GridSize,GridSize)

        self.neighborkey = [np.array(GetNeighbors(l,self.GridSize)) for l in range(self.GlobeGrid[0].size)]

    def AddMountains(self,NumMts = 50,mtHeight = 0.5,dx = 0.01,minlen=0.5,maxlen=2,wigglyness = 1):
        self.Elevation, self.LandIndicator = AddMts(self,NumMts = NumMts,mtHeight = mtHeight,dx = dx,minlen=minlen,maxlen=maxlen,wigglyness = wigglyness)
