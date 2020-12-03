import numpy as np
from matplotlib import cm


class Race:

    def __init__(self,world,name,parent = "Originator",color = 0,type = "Human", movementspd = 10,t0 = 0, growthrate = 0.4,deathrate = 1,renewabledepletion = 0.01,nonrenewabledepletion = 0.001,startRain = 0.7):

        self.Name = name
        self.Type = type
        self.Parent = parent

        PossibleStarts = np.where((world.Elevation > world.oLevel)*(world.RainFall > startRain))
        if len(PossibleStarts[0]):
            pk = np.random.randint(len(PossibleStarts[0]))
        else:
            PossibleStarts = np.where((world.Elevation > world.oLevel))
            pk = np.random.randint(len(PossibleStarts[0]))
        thept = PossibleStarts[0][pk]*world.GlobeGrid[0].shape[0] + PossibleStarts[1][pk]

        InitialPop = np.zeros(len(world.gridindices))
        InitialPop[thept] = 1
        self.InitialDistribution = InitialPop.reshape(world.GlobeGrid[0].shape)
        self.Population = self.InitialDistribution.copy()

        grs = growthrate*np.ones(len(InitialPop))
        grs[world.OceanIndicator]  = 0
        self.growthrates = grs.reshape(world.GlobeGrid[0].shape)

        drs = deathrate*np.ones(len(InitialPop))
        drs[world.OceanIndicator]  = 0
        self.deathrates = drs.reshape(world.GlobeGrid[0].shape)

        rds = renewabledepletion*np.ones(len(world.InitialRenew.flatten()))
        rds[world.OceanIndicator]  = 0
        self.RenewDeplete = rds

        nrds = nonrenewabledepletion*np.ones(len(world.InitialRenew.flatten()))
        nrds[world.OceanIndicator]  = 0
        self.NonRenewDeplete = nrds

        self.Movement = movementspd
        #if color == 'r':
        if hasattr(color, "__len__"):
            if len(color) == 3:
                self.BaseChromosome = color
            else:
                self.BaseChromosome = np.array(cm.nipy_spectral(np.random.rand())[:3])#np.random.rand(3)
        else:
            self.BaseChromosome = np.array(cm.nipy_spectral(np.random.rand())[:3])#np.random.rand(3)


        self.Chromosomes = self.BaseChromosome*self.InitialDistribution.astype(bool)[:,:,None]

        self.Cities = np.zeros_like(self.Population)

        self.History = {"Population":[self.InitialDistribution.copy()], "Genetics":[self.Chromosomes.copy()],"Cities":[self.Cities], "Time":[t0]}
