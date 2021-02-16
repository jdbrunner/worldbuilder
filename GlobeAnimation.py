import PIL
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from FantasyWorld import *
from matplotlib import cm
from matplotlib import colors as cmod
from scipy.interpolate import interp2d
from matplotlib.animation import FuncAnimation, PillowWriter
from plotting import *
import PIL
import io

def MakeColArr(Elv,LI,newL):
    colorArr = np.zeros((newL[1],newL[0],4))
    colorArr[LI.T] = cm.YlGn_r(Elv.T)[LI]
    colorArr[np.invert(LI.T)] = cm.Blues_r(Elv.T)[np.invert(LI.T)]
    return colorArr

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def turnImg(fig,ax,angle):

    ax.view_init(elev = 0, azim = angle)
    img = fig2img(fig)

    return  img


def MakeGif(world,SaveDir,num_frames = 10, spinTime = 5):#
    #need to increase the resolution of the elevation array so it doesn't look like potato
    n = 10
    Phi = world.GlobeGrid[0][0]
    Theta = world.GlobeGrid[1].T[0]
    ElevationInterp = interp2d(Phi,Theta,world.Elevation)

    newLen = (n*len(Phi),n*len(Theta))

    NewPhi = np.linspace(0, np.pi, newLen[0])
    NewTheta = np.linspace(-np.pi,np.pi,newLen[1])
    ElvDetail = ElevationInterp(NewPhi,NewTheta)
    DetailLI = ElvDetail > world.oLevel

    colorArr = MakeColArr(ElvDetail,DetailLI,newLen)

    # coordinates of the image - don't know if this is entirely accurate, but probably close
    lons = np.linspace(-180, 180, colorArr.shape[1]) * np.pi/180
    lats = np.linspace(-90, 90, colorArr.shape[0])[::-1] * np.pi/180

    # repeat code from one of the examples linked to in the question, except for specifying facecolors:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)
    x = np.outer(np.cos(lons), np.cos(lats)).T
    y = np.outer(np.sin(lons), np.cos(lats)).T
    z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    ax.set_axis_off()
    Grid = np.meshgrid(NewPhi,NewTheta)
    grdSize = Grid[0].size
    resol = 200
    ax.plot_surface(x, y, z, facecolors = colorArr,rcount = resol,ccount = resol)

    print("Making Globe Animation")
    frms = [turnImg(fig,ax,th) for th in np.linspace(0,360,num_frames)]
    print("Saving Animation")
    frms[0].save(SaveDir + '/SpinningGlobe.gif',save_all=True, append_images=frms[1:], optimize=False, duration=spinTime,loop=0)
    print("Animation Saved")

    # static.save("Static.png")
    #
    # print("Making Globe Animation")
    # ani = FuncAnimation(fig, updatefig,frames = np.linspace(0,360,num_frames))
    # print("Saving Animation")
    # writer = PillowWriter(fps=frpsec)
    # ani.save(SaveDir + "/SpinningGlobe.gif", writer=writer)
    #
    # print("Animation Saved")

if __name__ == "__main__":
    GS =100
    World1 = FantasyWorld(GridSize = GS,oLevel = 0.4)
    print("World Made")
    MakeGif(World1,num_frames = 100)
