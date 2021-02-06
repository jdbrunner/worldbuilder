import numpy as np
from geofuns import *

def SimpleWind(ph,th):
    try:
        return np.sin(ph)*(np.array([np.sin(th),-np.cos(th),0*th]) + 1*(np.array([0*th,0*th,np.cos(ph)]) - (np.cos(ph)**2)*toCart(ph,th,np.ones(ph.shape))))
    except:

        a = np.array([np.sin(th),-np.cos(th),0])

        b = np.array([0,0,np.cos(ph)])

        c =  (np.cos(ph)**2)*np.array(toCart(ph,th,1))



        return np.sin(ph)*(a + b - c)



def coastalness(p,t,ocean,windFun,coastdecay = 8,coasteffect = 1):

    cartGlobe = toCart(p,t,np.ones(p.shape))
    coast = np.zeros(p.shape)
    for pt in ocean:

        dis = GCdist((p,t),(pt[0],pt[1]))

        cartpt = toCart(*pt,1)
        windDir = windFun(*pt)

        cartDir = (-cartpt[0]+cartGlobe[0],-cartpt[1]+cartGlobe[1],-cartpt[2]+cartGlobe[2])
        mag = np.sqrt(cartDir[0]**2 + cartDir[1]**2 + cartDir[2]**2)
        cartDir = np.divide(cartDir,mag, out = np.zeros_like(cartDir),where=mag>0)

        windEf = windDir[0]*cartDir[0] + windDir[1]*cartDir[1] + windDir[2]*cartDir[2]

        effDis = np.divide(dis,windEf + 1, out = dis, where = windEf>0)

        coast += np.exp(-coastdecay*effDis)


    return coast



def baseTemp(world, m = 0.3, winterVar = 3, summerVar = 2, extremes = 0.05, coastdecay = 8,coasteffect = 1):
    '''p(latitude: 0-pi),t(longitude -pi-pi), W(Elevation,oceanlocation), oLev(ocean level: 0-1).
    m is the effect of elevation - m=0 means that elevation has no effect, m>0 means elevation linearly decreases temp
    by m*(hot-cold) at the highest point.

    Returns temp relative to a global max/min. Thus, 0.1 is 0.1*(global max - global min) + global min. Furthermore, at extreme
    latitude and elevation, we may have temperatures below global max.
    '''

    El = world.Elevation
    Oc = world.oceans

    winterX = 2*winterVar*world.GlobeGrid[0]/np.pi  - winterVar
    baseWinter = np.exp(-(winterX)**2)*(1-extremes)
    summerX = 2*summerVar*world.GlobeGrid[0]/np.pi  - summerVar
    baseSummer = np.exp(-(summerX)**2)*(1-extremes) + extremes

    elevWinter = baseWinter - m*(El-world.oLevel)/(1-world.oLevel)
    elevSummer = baseSummer - m*(El-world.oLevel)/(1-world.oLevel)

    elevRange = elevSummer - elevWinter

    coastal = coastalness(*world.GlobeGrid,Oc,world.Wind,coastdecay = coastdecay,coasteffect =coasteffect)
    coastal = coastal/np.max(coastal)

    Winter = elevWinter + (elevRange*coastal/2)*0.7
    Summer = elevSummer - (elevRange*coastal/2)*0.7

    return (Winter,Summer)


def rainFall(world, m = 0.2, coastdecay = 7,coasteffect = 0.4):
    '''p(latitude: 0-pi),t(longitude -pi-pi), W(Elevation,oceanlocation), oLev(ocean level: 0-1).
    m is the effect of elevation - m=0 means that elevation has no effect, m>0 means elevation linearly decreases temp
    by m*(hot-cold) at the highest point.

    Returns temp relative to a global max/min. Thus, 0.1 is 0.1*(global max - global min) + global min. Furthermore, at extreme
    latitude and elevation, we may have temperatures below global max.
    '''

    El = world.Elevation
    ocean = world.oceans

    base = 0.5*np.ones(world.GlobeGrid[0].shape)

    elev = base - m*(El-world.oLevel)/(1-world.oLevel)

    cartGlobe = toCart(*world.GlobeGrid,np.ones(world.GlobeGrid[0].shape))
    coast = np.zeros(world.GlobeGrid[0].shape)
    for pt in ocean:

        dis = GCdist((world.GlobeGrid[0],world.GlobeGrid[1]),(pt[0],pt[1]))

        cartpt = toCart(*pt,1)
        windVec = world.Wind(*pt)
        windMag = np.sqrt(windVec[0]**2 + windVec[1]**2 + windVec[2]**2)
        windDir = np.divide(windVec,windMag,out = np.zeros_like(windVec),where=windMag>0)

        cartDir = (cartpt[0]-cartGlobe[0],cartpt[1]-cartGlobe[1],cartpt[2]-cartGlobe[2])
        mag = np.sqrt(cartDir[0]**2 + cartDir[1]**2 + cartDir[2]**2)
        cartDir = np.divide(cartDir,mag, out = np.zeros_like(cartDir),where=mag>0)

        windEf = windDir[0]*cartDir[0] + windDir[1]*cartDir[1] + windDir[2]*cartDir[2]

        coast += windEf*np.exp(-coastdecay*dis)

    coast = coast - np.min(coast)
    coast = coast/np.max(coast)

    uncropped = elev + coasteffect*coast

    rain = uncropped - np.min(uncropped)
    rain = rain/np.max(rain)



    return rain


def growRiver(pt,oceans,rivers,world):
    #get surrounding points
    pts = [((pt[0]+i) % world.GlobeGrid[0].shape[0],(pt[1]+j) %  world.GlobeGrid[0].shape[1]) for i in [-1,0,1] for j in [-1,0,1]]
    nxtpt = pts[np.argmin([world.Elevation[p] for p in pts])]
    if nxtpt in rivers:
        return -1
    elif nxtpt in oceans:
        return -1
    else:
        return nxtpt

def makeRivers(world,riverProb = 0.1):

    ocean_indices = list(zip(*np.where(np.invert(world.LandIndicator))))
    wellSprings = np.random.rand(*world.RainFall.shape) < riverProb*world.RainFall
    wellSprings[np.invert(world.LandIndicator)] = False

    wellSpringsIndices = list(zip(*np.where(wellSprings)))
    river_indices = list(zip(*np.where(wellSprings)))

    for ws in wellSpringsIndices:
        npt = ws
        while npt != -1:
            npt = growRiver(npt,ocean_indices,river_indices,world)
            if npt !=-1:
                river_indices += [npt]

    river_locs = [(world.GlobeGrid[0][pt],world.GlobeGrid[1][pt]) for pt in river_indices]

    return river_indices,river_locs


def renew0(i,j,world,max_river_bonus = 0.4):
    gpsizes = dotArea(world.GlobeGrid[0],world.Radius,world.GridSize[1])
    gsize = gpsizes[i,j]/np.max(gpsizes)
    rain = world.RainFall[i,j]
    temp = np.arctan((world.Temps[0][i,j] + world.Temps[1][i,j])-1)/np.pi + 1
    if (i,j) in world.RiverIndices:
        return (rain*temp + (0.1+max_river_bonus)*np.random.rand())*gsize
    else:
        return (rain*temp + 0.1*np.random.rand())*gsize


def initNonR(i,j,world):
    gpsizes = dotArea(world.GlobeGrid[0],world.Radius,world.GridSize[1])
    gsize = gpsizes[i,j]/np.max(gpsizes)
    base = 2*(np.arctan(2*(world.Elevation[i,j] - 0.5))/np.pi + np.arctan(1)/np.pi)#(0.6*np.arctan(2*(world.Elevation[i,j] - 0.5))/np.pi)
    latscl = 0.4*np.sin(world.GlobeGrid[0][i,j]) + 0.6
    return (base*np.random.rand()*latscl)*gsize
