import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors as cmod
import subprocess as sub

from scipy.integrate import ode

from scipy.interpolate import interp2d

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from IPython.display import HTML

import matplotlib.patches as mpatches

import pandas as pd
from matplotlib.animation import PillowWriter

import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

from geofuns import *
from FantasyWorld import *
from plotting import *
from demographics import *


def ShowSquare(upperleft,lowerright,world):
    MapTheta = mapThetas(world.GlobeGrid)
    RiverMsk = world.RiverIndicator.reshape(world.GridSize)

    fig = plt.figure(figsize = (10,5))
    ax = plt.axes()
    ax.set_ylim(np.pi,0)

    ax.scatter(world.GlobeGrid[1][world.LandIndicator],world.GlobeGrid[0][world.LandIndicator],s=20,c = world.Elevation[world.LandIndicator], cmap = 'YlGn_r')
    ax.scatter(world.GlobeGrid[1][np.invert(world.LandIndicator)],world.GlobeGrid[0][np.invert(world.LandIndicator)],s=20,c = world.Elevation[np.invert(world.LandIndicator)], cmap = 'Blues_r')
    ax.scatter(world.GlobeGrid[1][RiverMsk],world.GlobeGrid[0][RiverMsk],s=5,color = 'b')

    line1 = [[upperleft[1] for i in range(100)],np.linspace(upperleft[0],lowerright[0],100)]
    ax.scatter(line1[0],line1[1],color = 'r')
    line2 = [[lowerright[1] for i in range(100)],np.linspace(upperleft[0],lowerright[0],100)]
    ax.scatter(line2[0],line2[1],color = 'r')
    line3 = [np.linspace(upperleft[1],lowerright[1],100), [upperleft[0] for i in range(100)]]
    ax.scatter(line3[0],line3[1],color = 'r')
    line4 = [np.linspace(upperleft[1],lowerright[1],100), [lowerright[0] for i in range(100)]]
    ax.scatter(line4[0],line4[1],color = 'r')

    plt.show()

    return fig,ax

def newrpts(a,direc,t):
    x = a[0] + t*direc[0] + (0.1*np.random.rand() + 0.05*np.sin(2*np.pi*t))*direc[1]
    y = a[1] + t*direc[1] - (0.1*np.random.rand() + 0.05*np.sin(2*np.pi*t))*direc[0]
    return (x,y)

def CreateMeander(a,b,res = 100):
    direc = (b[0]-a[0],b[1]-a[1])
    ln = [newrpts(a,direc,t) for t in np.linspace(0,1,res)]
    return ln

def LocalRivers(upperleft,lowerright,world,ElevationInterp):
    local_rivers = [rv for rv in world.RiverLocs if (rv[0]<lowerright[0] and rv[0]>upperleft[0] and rv[1]<lowerright[1] and rv[1]>upperleft[1])]
    rivsphi = [rv[0] for rv in local_rivers]
    rivthet = [rv[1] for rv in local_rivers]

    rivers = []
    unassigned = local_rivers
    while len(unassigned):
        strt = unassigned[0]
        nearest = [pt for pt in unassigned if GCdist(pt,strt) < np.pi/30]
        river = nearest
        unassigned = [pt for pt in unassigned if pt not in river]
        while len(nearest):
            nearest =[pt for pt in unassigned if GCdist(pt,river[-1]) < np.pi/30]
            river += nearest
            unassigned = [pt for pt in unassigned if pt not in river]
        rivers += [river]

    for riv in rivers:
        if ElevationInterp(*riv[-1]) - world.oLevel > 0.01:
            h = 0.001
            dx = (1/(2*h))*(ElevationInterp(riv[-1][0] + h,riv[-1][1]) - ElevationInterp(riv[-1][0] - h,riv[-1][1]))
            dy = (1/(2*h))*(ElevationInterp(riv[-1][0],riv[-1][1] + h) - ElevationInterp(riv[-1][0],riv[-1][1] - h))
            newpt = (riv[-1][0] - 0.01*dx,riv[-1][1] - 0.01*dy)
            fel = ElevationInterp(*newpt) - world.oLevel
            msteps = 20
            stps = 0
            while fel > 0.01 and stps < msteps:
                newpt = (newpt[0] - 0.01*dx,newpt[1] - 0.01*dy)
                fel = ElevationInterp(*newpt) - world.oLevel
                stps += 1
            riv += [newpt]
    rivers2 = []
    for riv in rivers:
        riv2 = []
        for i in range(len(riv) - 1):
            riv2 += [CreateMeander(riv[i],riv[i+1],res = 200)]
        rivers2 += riv2


    return rivers2

def nearest_gp(coord,mshgrid):
    if coord[0]>=mshgrid[0][0][0]:
        lowerX = np.where(mshgrid[0][0]<=coord[0])[0][-1]
    else:
        lowerX = 0
    if coord[0]<=mshgrid[0][0][-1]:
        upperX = np.where(mshgrid[0][0]>=coord[0])[0][0]
    else:
        upperX = len(mshgrid[0][0]) - 1
    if coord[1]>=mshgrid[1][0,0]:
        lowerY = np.where(mshgrid[1][:,0]<=coord[1])[0][-1]
    else:
        lowerY = 0
    if coord[1]<=mshgrid[1][-1,0]:
        upperY = np.where(mshgrid[1][:,0]>=coord[1])[0][0]
    else:
        upperY = len(mshgrid[1][:,0]) - 1
    c1 = (mshgrid[0][lowerY,lowerX],mshgrid[1][lowerY,lowerX])
    c2 = (mshgrid[0][lowerY,upperX],mshgrid[1][lowerY,upperX])
    c3 = (mshgrid[0][upperY,lowerX],mshgrid[1][upperY,lowerX])
    c4 = (mshgrid[0][upperY,upperX],mshgrid[1][upperY,upperX])
    dists = [GCdist(pt,coord) for pt in [c1,c2,c3,c4]]
    return [(lowerY,lowerX),(lowerY,upperX),(upperY,lowerX),(upperY,upperX)][np.argmin(dists)]

def LocalPol(LocalLand,coarsepol,coarseGrd,P,T):
    country_detail = np.zeros((LocalLand.shape[0] + 2,LocalLand.shape[1] + 2))
    for i in range(len(coarseGrd[0])):
        for j in range(len(coarseGrd[1].T)):
            smlc = nearest_gp((coarseGrd[0][i,j],coarseGrd[1][i,j]),(P,T))
            country_detail[smlc] = coarsepol[i,j]

    LandPadded = np.concatenate([[[False]*LocalLand.shape[1]],LocalLand,[[False]*LocalLand.shape[1]]])
    LandPadded = np.concatenate([np.array([[False]*(LocalLand.shape[1]+2)]).T,LandPadded,np.array([[False]*(LocalLand.shape[1]+2)]).T],axis = 1)

    numfld = np.sum(country_detail[LandPadded].astype(bool))
    while (not np.all(country_detail[LandPadded])):
        for i in range(1,country_detail.shape[0]-1):
            for j in range(1,country_detail.shape[1]-1):
                if LandPadded[i,j]:
                    if country_detail[i,j] == 0:
                        neighbors = [country_detail[i+l,j+k] for k in [-1,0,1] for l in [-1,0,1]]
                        country_detail[i,j] = np.random.choice(neighbors)
        numfld2 = np.sum(country_detail[LandPadded].astype(bool))
        if numfld2 == numfld:
            break
        else:
            numfld = numfld2
    country_detail = country_detail[1:-1,1:-1]
    return country_detail

def Cities(upperleft,lowerright,LocalLand,world,demos,race):
    totalDens = demos.History[race][-1]
    Phi = world.GlobeGrid[0][0]
    Theta = world.GlobeGrid[1].T[0]
    PopInterp = interp2d(Phi,Theta,totalDens, kind = 'quintic')
    phi = np.linspace(upperleft[0],lowerright[0],500)
    theta = np.linspace(upperleft[1],lowerright[1],500)
    P,T = np.meshgrid(phi,theta)
    LocalPop = PopInterp(phi,theta)
    LocalPop[np.invert(LocalLand)] = 0
    neighborhood_size = 10

    data_max = filters.maximum_filter(LocalPop, neighborhood_size)
    maxima = (LocalPop == data_max)
    maxima[np.invert(LocalLand)] = 0
    return maxima

def ZoomInRegion(upperleft,lowerright,world,demographics,res = 500):
    Phi = world.GlobeGrid[0][0]
    Theta = world.GlobeGrid[1].T[0]
    ElevationInterp = interp2d(Phi,Theta,world.Elevation, kind = 'quintic')
    phi = np.linspace(upperleft[0],lowerright[0],res)
    theta = np.linspace(upperleft[1],lowerright[1],res)
    P,T = np.meshgrid(phi,theta)
    Local = ElevationInterp(phi,theta)
    LocalLand = Local>world.oLevel

    startphi = np.where(world.GlobeGrid[0][0]>upperleft[0])[0][0]
    stopphi = np.where(world.GlobeGrid[0][0]>=lowerright[0])[0][0]
    startthe = np.where(world.GlobeGrid[1][:,0]>upperleft[1])[0][0]
    stopthe = np.where(world.GlobeGrid[1][:,0]>=lowerright[1])[0][0]


    local_rivers = LocalRivers(upperleft,lowerright,world,ElevationInterp)


    coarseGrd = (world.GlobeGrid[0][startthe:stopthe,startphi:stopphi],world.GlobeGrid[1][startthe:stopthe,startphi:stopphi])

    coarsepol = demographics.PoliticalHistory[-1][startthe:stopthe,startphi:stopphi].astype(int)
    LocalCountry = LocalPol(LocalLand,coarsepol,coarseGrd,P,T)

    cities = Cities(upperleft,lowerright,LocalLand,world,demographics,'total')

    return Local,LocalLand,local_rivers,LocalCountry,cities,P,T

def GetDemographics(Country,demographics, t = -1):
    pops = {}
    for i in range(len(demographics.Races)):
        pops[demographics.Races[i]] = np.sum(demographics.Populations[i][demographics.PoliticalHistory[t] == Country])
    return pd.Series(pops)

def GetDemosRegion(CountryList,demographics, t = -1):
    retDf = pd.DataFrame(index = demographics.Races)
    retDf["Total"] = np.zeros(len(retDf))
    for c in CountryList:
        ser =  GetDemographics(c,demographics,t = t)
        retDf[c] = ser
        retDf["Total"] += ser
    return retDf


class Region:

    def __init__(self,upperleft,lowerright,world,demographics,res = 500):
        Local,LocalLand,local_rivers,LocalCountry,cities,P,T = ZoomInRegion(upperleft,lowerright,world,demographics,res=res)
        self.Phi = P
        self.Theta = T
        self.Elevation = Local
        self.Land = LocalLand
        self.Rivers = local_rivers
        self.Countries = LocalCountry
        self.CityMasks = cities

        self.PopInterp = interp2d(world.GlobeGrid[0][0],world.GlobeGrid[1].T[0],demographics.History["total"][-1], kind = 'quintic')

        self.Cities = self.ListCities()

        self.width = GCdist(((upperleft[0]+lowerright[0])/2,upperleft[1]),((upperleft[0]+lowerright[0])/2,lowerright[1]))
        self.height = GCdist((upperleft[0],(upperleft[1]+lowerright[1])/2),(lowerright[0],(upperleft[1]+lowerright[1])/2))

        self.DemographicStats = GetDemosRegion(np.delete(np.unique(self.Countries),0),demographics)

        self.CountryNames = {}

    def ListCities(self):
        citylocs =  (self.Phi[np.where(self.CityMasks)],self.Theta[np.where(self.CityMasks)])
        cityDF = pd.DataFrame(index = ['C'+str(i) for i in range(len(citylocs[0]))], columns = ["Location","Population","Name"])
        for i in range(len(citylocs[0])):
            cityDF.loc['C' + str(i)] = [(citylocs[0][i],citylocs[1][i]),self.PopInterp(citylocs[0][i],citylocs[1][i])[0],'']
        # citylist = [('C' + str(i),(citylocs[0][i],citylocs[1][i]),self.PopInterp(citylocs[0][i],citylocs[1][i])) for i in range(len(citylocs[0]))]
        return cityDF

    def ShowMap(self,left = 0,right = 1,bottom = 0,top = 1,labelCities = False, cityLabel = "Name", LabelCountries = False, countryLabel = "Name"):

        leftside = (1-left)*self.Theta[0,0] + left*self.Theta[-1,0]
        rightside = (1-right)*self.Theta[0,0] + right*self.Theta[-1,0]

        topside = (1-top)*self.Phi[0,-1] + top*self.Phi[0,0]
        bottomside = (1-bottom)*self.Phi[0,-1] + bottom*self.Phi[0,0]

        fscl = (top-bottom)

        ratio = GCdist(((topside+bottomside)/2,leftside),((topside+bottomside)/2,rightside))/GCdist((topside,(rightside+leftside)/2),(bottomside,(rightside+leftside)/2))

        fig = plt.figure(figsize = (20*ratio,20))
        ax = plt.axes()

        ax.set_ylim(bottomside,topside)
        ax.set_xlim(leftside,rightside)

        ax.scatter(self.Theta[self.Land],self.Phi[self.Land],c = self.Elevation[self.Land], cmap = 'YlGn_r', alpha = 0.5)
        ax.scatter(self.Theta[np.invert(self.Land)],self.Phi[np.invert(self.Land)],c= self.Elevation[np.invert(self.Land)],cmap = 'Blues_r')
        for riv in self.Rivers:
            ax.plot([r[1] for r in riv],[r[0] for r in riv],c = 'b',linewidth=5, alpha = 0.5)
        for ci in np.unique(self.Countries):
            if ci:
                cionly = (self.Countries == ci).astype(int)
                ax.contour(self.Theta,self.Phi,cionly,levels = 1,colors = 'tab:orange',linewidths = 2)

        ax.scatter(self.Theta[self.CityMasks],self.Phi[self.CityMasks],c = 'r',s = 20/fscl)
        if labelCities:
            for ci in self.Cities.index:
                loca = self.Cities.loc[ci,"Location"]
                if (loca[1] < rightside and loca[1]>leftside and loca[0] > topside and loca[0]<bottomside):
                    if cityLabel == "Name" and len(self.Cities.loc[ci,"Name"]):
                        ax.text(self.Cities.loc[ci,"Location"][1],self.Cities.loc[ci,"Location"][0],self.Cities.loc[ci,"Name"],fontsize=20/fscl,color = 'k')
                    elif cityLabel == "Both" and len(self.Cities.loc[ci,"Name"]):
                        ax.text(self.Cities.loc[ci,"Location"][1],self.Cities.loc[ci,"Location"][0],ci+":" +self.Cities.loc[ci,"Name"],fontsize=20/fscl,color = 'k')
                    else:
                        ax.text(self.Cities.loc[ci,"Location"][1],self.Cities.loc[ci,"Location"][0],ci,fontsize=20/fscl,color = 'k')

        if LabelCountries:
            for ci in np.delete(np.unique(self.Countries),0):
                alllocs = np.where((self.Countries == ci).astype(int))
                tot = len(alllocs[0])
                whch = int(tot/2)
                pt = (self.Theta[alllocs[0][whch],alllocs[1][whch]],self.Phi[alllocs[0][whch],alllocs[1][whch]])
                if (pt[0] < rightside and pt[0]>leftside and pt[1] > topside and pt[1]<bottomside):
                    if countryLabel == "Name" and ci in self.CountryNames.keys():
                        ax.text(*pt,self.CountryNames[ci],fontsize=25/fscl,color = 'navy')
                    elif countryLabel == "Both" and ci in self.CountryNames.keys():
                        ax.text(*pt,ci + ":" + self.CountryNames[ci],fontsize=25/fscl,color = 'navy')
                    else:
                        ax.text(*pt,ci,fontsize=25/fscl,color = 'navy')

        return fig,ax



    def ShowCountry(self,country):
        leftside = self.Theta[0,0]
        rightside = self.Theta[-1,0]

        topside = self.Phi[0,0]
        bottomside = self.Phi[0,-1]

        ratio = GCdist(((topside+bottomside)/2,leftside),((topside+bottomside)/2,rightside))/GCdist((topside,(rightside+leftside)/2),(bottomside,(rightside+leftside)/2))

        fig = plt.figure(figsize = (20*ratio,20))
        ax = plt.axes()

        ax.set_ylim(bottomside,topside)
        ax.set_xlim(leftside,rightside)

        ax.scatter(self.Theta[self.Land],self.Phi[self.Land],c = self.Elevation[self.Land], cmap = 'YlGn_r')
        ax.scatter(self.Theta[np.invert(self.Land)],self.Phi[np.invert(self.Land)],c= self.Elevation[np.invert(self.Land)],cmap = 'Blues_r')
        for riv in self.Rivers:
            ax.plot([r[1] for r in riv],[r[0] for r in riv],c = 'b',linewidth=5)
        for ci in np.unique(self.Countries):
            if ci:
                cionly = (self.Countries == ci).astype(int)
                ax.contour(self.Theta,self.Phi,cionly,levels = 1,colors = 'k',linewidths = 2)
        ax.scatter(self.Theta[self.Countries == country],self.Phi[self.Countries == country],alpha = 0.2)

        return fig

    def NameCities(self):
        with open("townnames.txt") as fl:
            lines = fl.readlines()
        townNames = [nm.replace("\n",'') for nm in lines]
        randomNames = np.random.choice(townNames,size = len(self.Cities),replace = False)
        for i in range(len(self.Cities)):
            self.Cities.loc["C" + str(i),"Name"] = randomNames[i]

    def NameCountries(self):
        with open("statenames.txt") as fl:
            lines = fl.readlines()
        statenames = [nm.replace("\n",'') for nm in lines]
        countrs = np.delete(np.unique(self.Countries),0)
        randomNames = np.random.choice(statenames,size = len(countrs),replace = False)
        for i in range(len(countrs)):
            self.CountryNames[countrs[i]] = randomNames[i]
