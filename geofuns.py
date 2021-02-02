import numpy as np


def toCart(p,t,r):
    return r*np.cos(t)*np.sin(p),r*np.sin(t)*np.sin(p),r*np.cos(p)

def toSphere(x,y,z):
    xpy = x**2 + y**2
    if xpy == 0:
        if z>0:
            r = z
            phi = 0
            theta = 0
        else:
            r = - z
            phi = np.pi
            theta = 0
    elif y == 0:
        r = np.sqrt(xpy + z**2)
        phi = np.arctan(z/np.sqrt(xpy))
        if x > 0:
            theta = 0
        else:
            theta = np.pi
    else:
        r = np.sqrt(xpy + z**2)
        phi = np.arctan(z/np.sqrt(xpy))
        theta = np.arctan(y/x)
    return r, phi,theta


def GCdist(pt1,pt2):
    '''Compute the great circle GCdistance between two points, returns as fraction of the planet radius'''
    x1,y1,z1 = toCart(pt1[0],pt1[1],1)
    x2,y2,z2 = toCart(pt2[0],pt2[1],1)
    cosSig = (x1*x2+y1*y2+z1*z2)
    sig = np.arccos(cosSig, out=np.zeros_like(cosSig), where=cosSig<=1)
    return sig

def MakeSpines(NumPlates,dx = 0.01,minlen=0.5,maxlen=2,wigglyness = 1):
    PlateBackbones = []
    for i in range(NumPlates):
        start = np.array([(0.6*np.random.rand()+0.2)*np.pi, np.random.rand()*2*np.pi])
        bbone = [start]
        length = np.random.randint(int(minlen/dx),int(maxlen/dx))
        angle = np.random.rand()*2*np.pi
        for j in range(length):
            angle = angle + wigglyness*np.random.rand() - wigglyness/2
            inc = (dx/np.pi)*np.array([np.cos(angle),np.sin(angle)])
            dxx = 0
            tmpPt = bbone[-1]
            while dxx < dx:
                tmpPt = tmpPt + inc
                dxx = GCdist(bbone[-1],tmpPt)
            bbone = bbone + [tmpPt]
        PlateBackbones = PlateBackbones + [[[bb[0] for bb in bbone],[bb[1] for bb in bbone],[np.random.rand() for bb in bbone]]]
    return PlateBackbones

def elevation(p,t,BBones,smth = 0.2,jagged = 0.1):
    el = 0
    for bb in BBones:
        for j in range(len(bb[0])):
            dis = GCdist((p,t),(bb[0][j],bb[1][j]))
            el += bb[2][j]*(1/(smth*2*np.pi))*np.exp(-(1/2)*((dis/smth)**2))
    return el*(1 + jagged*np.random.rand())

def MakeContinents(world,NumPlates = 50,dx = 0.01,minlen=0.5,maxlen=2,wigglyness = 1,smth = 0.2,jagged = 0.1):
    PlateBackbones = MakeSpines(NumPlates,dx = dx, minlen = minlen,maxlen = maxlen,wigglyness = wigglyness)
    El = elevation(*world.GlobeGrid,PlateBackbones,smth = smth, jagged = jagged)
    El = El/np.max(El)
    return El, El>world.oLevel

def AddMts(world,NumMts = 50,mtHeight = 0.5,dx = 0.01,minlen=0.5,maxlen=2,wigglyness = 1):
    spines = MakeSpines(NumMts,dx = dx, minlen = minlen,maxlen = maxlen,wigglyness = wigglyness)
    mts = 0
    grdSize = world.GlobeGrid[0].shape[0]
    ptdis = np.pi/100
    smth = ptdis
    for bb in spines:
        for j in range(len(bb[0])):
            dis = GCdist((world.GlobeGrid[0],world.GlobeGrid[1]),(bb[0][j],bb[1][j]))
            mts += bb[2][j]*(1/(smth*2*np.pi))*np.exp(-(1/2)*((dis/smth)**2))
    mts = mts/np.max(mts)
    elev = mts*mtHeight + world.Elevation
    return elev, elev > world.oLevel

def dotArea(lat,radius,gs):
    return np.pi*(radius*np.pi/(gs-1))*(radius*np.cos(np.pi/2 - lat)/(gs-1))

def getDists(j,N,M,Dfun,Grid):
    return np.array([[Dfun((Grid[0][j],Grid[1][j]),(Grid[0][(j[0] + l) % N, max(0,min(j[1]+m,M-1))],Grid[1][(j[0] + l) % N,  max(0,min(j[1]+m,M-1))]))for m in [-1,0,1]] for l in [-1,0,1]])

def getSurrounding(j,D):
    N,M = D.shape[:2]
    return np.array([[D[((j[0] + l) % N, max(0,min((j[1]+m),M-1)))] for m in [-1,0,1]] for l in [-1,0,1]])
