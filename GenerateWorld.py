import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors as cmod
import subprocess as sub
import os
import sys

from scipy.integrate import ode

from scipy.interpolate import interp2d

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from IPython.display import HTML

import matplotlib.patches as mpatches


from matplotlib.animation import PillowWriter


from scipy.spatial.transform import Rotation as R

from geofuns import *
from FantasyWorld import *
from plotting import *
from demographics import *

import pickle

NameOfFolder = sys.argv[1]

try:
    os.mkdir(NameOfFolder)
except:
    pass



GS =100
World1 = FantasyWorld(GridSize = GS,oLevel = 0.4)

MapTheta = mapThetas(World1.GlobeGrid)

RiverMsk = World1.RiverIndicator.reshape(World1.GridSize)

fig = plt.figure(figsize = (10,5))
ax = plt.axes()
ax.set_ylim(np.pi,0)
ax.scatter(MapTheta[World1.LandIndicator],World1.GlobeGrid[0][World1.LandIndicator],s=20,c = World1.Elevation[World1.LandIndicator], cmap = 'YlGn_r')
ax.scatter(MapTheta[np.invert(World1.LandIndicator)],World1.GlobeGrid[0][np.invert(World1.LandIndicator)],s=20,c = World1.Elevation[np.invert(World1.LandIndicator)], cmap = 'Blues_r')
ax.scatter(MapTheta[RiverMsk],World1.GlobeGrid[0][RiverMsk],s=5,color = 'b')
plt.savefig(NameOfFolder+"/WorldMap.png")

np.save(NameOfFolder+"/Elevation",World1.Elevation)
np.save(NameOfFolder+"/GlobeMesh",World1.GlobeGrid)
np.save(NameOfFolder+"/Temps",World1.Temps[1])

sub.run("python GlobeAnimation.py "+ NameOfFolder+"/GlobeMesh.npy " + NameOfFolder+"/Elevation.npy "+ NameOfFolder+ "/Temps.npy")

#Should probably have these load from file or something.
Kappas = 0.1*np.array([1.1,0.9,0.7,0.8,1.1]) #np.random.rand(numPops)
Gammas = 0.1*np.array([1,0.9,0.7,0.8,1.1]) #11*np.random.rand(numPops)
beta1s = 0.1*np.array([1,0.8,1,0.9,0.8]) #0.5*np.random.rand(numPops)
beta2s = 0.01*np.array([1,0.8,0.9,1,0.8]) #0.01*np.random.rand(numPops)
speeds = [2,1,0.9,0.8,0.9]
ThePops = ['humans','halfling','elves','dwarves','gnomes']
numPops = len(ThePops)
Starters = [World1.RiverIndices[np.random.choice(len(World1.RiverIndices))] for r in range(numPops)]
IntialPops = np.zeros((numPops,*World1.GridSize))
for st in range(len(Starters)):
    IntialPops[st,Starters[st][0],Starters[st][1]] = 0.1

demos = Demographics(ThePops,IntialPops,Kappas,Gammas,beta1s,beta2s,speeds)

#call it units of decades?
tstp = 1 #decade time-steps until we get farther on then switch to year. Any larger time-steps messes with migration/disasters.
ExpectedDisastersPerDecade = 1 #maximum of 1/dt
ExpectedMigrationsPerDecade = 0.5 #maximum of 1/dt

while demos.History["time"][-1] < 500:
    if demos.History["time"][-1] > 300:
        tstp = 0.5
    if demos.History["time"][-1] > 400:
        tstp = 0.2
    if any([np.sum(demos.History[pop][-1]).round(5) == 0 for pop in demos.Races]):
        print("Extinction:", pop)
    World1.Renewables,World1.NonRenewables = demos.ChangePop(World1,dt = tstp)
    #roll for a disaster:
    if np.random.rand() < ExpectedDisastersPerDecade*tstp:
        demos.NaturalDisaster()
    #roll for a migration:
    if np.random.rand() < (demos.History["time"][-1]/(100 + demos.History["time"][-1]))*ExpectedMigrationsPerDecade*tstp:
        demos.Migration(World1)
    ExpectedMigrationsPerDecade = min(0.1 + 0.1*demos.History["time"][-1],1) #maximum of 1/dt


fig = plt.figure(figsize = (10,5))
ax = plt.axes()
for rc in ThePops:
    ax.plot([10*t for t in demos.History["time"]],[np.sum(pop) for pop in demos.History[rc]], label = rc)
ax.legend()
plt.savefig(NameOfFolder+"/TotalPopulations.png")


fig,ax = plt.subplots(5,1,figsize = (10,25))

for axs in ax:
    axs.set_ylim(np.pi,0)
    axs.scatter(MapTheta[World1.LandIndicator],World1.GlobeGrid[0][World1.LandIndicator],s=20,c = World1.Elevation[World1.LandIndicator], cmap = 'YlGn_r')
    axs.scatter(MapTheta[np.invert(World1.LandIndicator)],World1.GlobeGrid[0][np.invert(World1.LandIndicator)],s=20,c = World1.Elevation[np.invert(World1.LandIndicator)], cmap = 'Blues_r')
    axs.scatter(MapTheta[RiverMsk],World1.GlobeGrid[0][RiverMsk],s=5,color = 'b')

nice_colors = [(0,0.8,1),(1,0,0.1),(0,1,0.1),(0.5,0,1),(1,1,0.1),(1,0.1,1)]

colors = {}
bascol = {}
ii=0
for pop in ThePops:
    c1,c2,c3 = nice_colors[ii]#np.random.rand(3)
    ii += 1
    bascol[pop] = (c1,c2,c3)
    colors[pop] =  LinearSegmentedColormap.from_list(pop+"mp", [(c1,c2,c3, 0.0),(c1,c2,c3, 0.5)])

for i in range(len(ThePops)):
    ax[i].scatter(MapTheta.flatten(),World1.GlobeGrid[0].flatten(),s=20,c  = demos.History[ThePops[i]][-1].flatten(),cmap =colors[ThePops[i]])
    ax[i].set_title(ThePops[i])

plt.savefig(NameOfFolder+"/PopulationDistributions.png")

fig = plt.figure(figsize = (10,5))
ax = plt.axes()
ax.set_ylim(np.pi,0)
tlmap = LinearSegmentedColormap.from_list('tlmap', [(0, 1, 1, 0.0),(0, 1, 1, 1.0)])


ax.scatter(MapTheta[World1.LandIndicator],World1.GlobeGrid[0][World1.LandIndicator],s=20,c = World1.Elevation[World1.LandIndicator], cmap = 'YlGn_r')
ax.scatter(MapTheta[np.invert(World1.LandIndicator)],World1.GlobeGrid[0][np.invert(World1.LandIndicator)],s=20,c = World1.Elevation[np.invert(World1.LandIndicator)], cmap = 'Blues_r')
ax.scatter(MapTheta[RiverMsk],World1.GlobeGrid[0][RiverMsk],s=5,color = 'b')
#
tpt = -1
ExistingColors = demos.PoliticalHistory[tpt].copy()
for ii,jj in demos.ExistingCountries.items():
    loc = np.where(demos.PoliticalHistory[tpt] == ii)
    ExistingColors[loc] = jj

im = ax.scatter(MapTheta.flatten()[ExistingColors.flatten()>0],World1.GlobeGrid[0].flatten()[ExistingColors.flatten()>0],s=20,c  = ExistingColors.flatten()[ExistingColors.flatten()>0],cmap = "gist_ncar")
plt.savefig(NameOfFolder+"/PoliticalMap.png")

pickle.dump(World1, open( NameOfFolder+"/world.p", "wb" ) )
pickle.dump(demos, open(NameOfFolder+ "/demographics.p", "wb" ) )
