import numpy as np
from scipy.integrate import ode
from geofuns import *
from scipy.spatial.transform import Rotation as R

def SqrPowers(pops,world):
    N_all = np.sum(world.LandIndicator)

    popDens = pops/dotArea(world.GlobeGrid[0],world.Radius,world.GridSize[1])

    avg_pop_all = np.sum(popDens)/N_all
    pop_var_all = np.sqrt(np.sum((popDens- avg_pop_all)**2)/N_all)
    above_all = (popDens - avg_pop_all)/pop_var_all

    final = above_all*world.LandIndicator

    final_floor = np.maximum(final,0.001*np.ones_like(final))

    return final_floor

def getPower(ci,countries,power):
    indices = np.where(countries == ci)
    return sum(power[indices])

def NewCountries(pop,current_countries,current_index,world,powers):
    new = ((powers > (3 + 3/(1 + 0.1*current_index)))*(current_countries == 0)).flatten()
    new_countries = current_countries.copy().flatten()
    newwhere = np.where(new)[0]
    for i in newwhere:
        neighbors = world.neighborkey[i]
        if all(current_countries.flatten()[neighbors] == 0):
            new_countries[neighbors] = current_index
            current_index += 1
    new_countries = new_countries.reshape(world.GridSize)*world.LandIndicator

    return new_countries,current_index

def ExpandCountry(ci,countries,world):
    new_countries = countries.copy().flatten()
    country = np.where(countries.flatten() == ci)[0]
    for i in country:
        neighbors = world.neighborkey[i]
        for j in neighbors:
            if (new_countries[j] == 0) or (new_countries[j] == ci):
                new_countries[j] = ci
    return new_countries.reshape(countries.shape)

def DisbandCountry(ci,countries):
    countries[np.where(countries == ci)] == 0
    return countries

def ViveLeRevolution(country,countries,currentIndex,world):
    #Need to figure out how to divide a country!!!
    new_countries = countries.copy()
    borderpts = GetPerim(country,new_countries,world)
    locs = np.where(new_countries == country)
    num_sqs = len(locs[0])
    num_rebelled = 1
    if num_sqs > 3 and len(borderpts):
        starting_point = np.random.choice(borderpts)
        new_countries[np.unravel_index(starting_point,world.GridSize)] = currentIndex
        num_sqs = num_sqs - 1
        for j in world.neighborkey[starting_point]:
            if new_countries.flatten()[j] == country:
                new_countries[np.unravel_index(j,world.GridSize)] = currentIndex
                num_sqs = num_sqs - 1
        continuing = True
        while continuing:
            if num_sqs > 0:
                border_points = findBorder(country,currentIndex,new_countries,world)
                if len(border_points):
                    for bp in border_points:
                        new_countries[np.unravel_index(bp,world.GridSize)] = currentIndex
                        num_sqs = num_sqs - 1
                    if np.random.rand() < 0.5:
                        continuing = False
                else:
                    continuing = False
            else:
                continuing = False
        currentIndex += 1
        return new_countries,currentIndex
    else:
        return new_countries,currentIndex

def GetNeighborCountries(country,countries,world):
    fltlocs = np.where(countries.flatten() == country)[0]
    flcountries = countries.copy().flatten()
    neighbor_countries = []
    for i in fltlocs:
        neighbors = world.neighborkey[i]
        for j in neighbors:
            if (flcountries[j] != 0) and (flcountries[j] != country):
                neighbor_countries += [flcountries[j]]
    return np.unique(neighbor_countries)

def WhatIsItGoodFor(country,countries,power,world):
    neighborcs = GetNeighborCountries(country,countries,world)
    new_countries = countries.copy()
    attacking = 0
    if len(neighborcs):
        locpower = getPower(country,countries,power)
        neighborpowers = [getPower(ci,countries,power) for ci in neighborcs]
        powerRatios = [locpower/(locpower + np) for np in neighborpowers]
        totR = sum(powerRatios)
        probs = [pr/totR for pr in powerRatios]
        attacking = np.random.choice(neighborcs, p = probs)
        locs = np.where(countries == attacking)
        dpow = getPower(attacking,countries,power)
        if dpow > 0.6*locpower:
            fighting = True
            while fighting:
                border_points = findBorder(attacking,country,new_countries,world)
                if len(border_points):
                    for bp in border_points:
                        if np.random.rand() > dpow/(locpower + dpow):
                            new_countries[np.unravel_index(bp,world.GridSize)] = country
                    if np.random.rand() < 0.5:
                        fighting = False
                else:
                    fighting = False
        else:
            new_countries[locs] = country
    return new_countries,attacking

def findBorder(c1,c2,countries,world):
    c1locs = np.where(countries.flatten() == c1)[0]
    border_pts = []
    for loc in c1locs:
        if c2 in countries.flatten()[world.neighborkey[loc]]:
            border_pts += [loc]
    return border_pts

def GetPerim(c1,countries,world):
    c1locs = np.where(countries.flatten() == c1)[0]
    border_pts = []
    for loc in c1locs:
        if any(countries.flatten()[world.neighborkey[loc]] != c1):
            border_pts += [loc]
    return border_pts

def GetCountryTotals(countries):
    countrySizes = {}
    for i in range(1,np.max(countries)+1):
        countrySizes[i] = len(np.where(countries == i)[0])
    existing_countries = [ky for ky in countrySizes.keys() if countrySizes[ky] > 0]
    ec_dict = dict([(existing_countries[i],i) for i in range(len(existing_countries))])
    total_existing = len(ec_dict)
    return countrySizes,ec_dict,total_existing


def GrowthSys(t,y,params):
    [kappas,gammas,alpha,R0,beta1s,beta2s,numpops] = params
    pops = y[:kappas.size]
    renews = y[kappas.size:kappas.size+alpha.size]
    nonrenews =  y[kappas.size+alpha.size:]
    popsDot = kappas*pops*np.array([renews]*numpops).flatten()*np.array([nonrenews]*numpops).flatten() - gammas*pops
    rdot = alpha*(R0-renews) - np.sum((beta1s*popsDot).reshape(numpops,alpha.size), axis = 0)
    ndot =  - np.sum((beta2s*popsDot).reshape(numpops,alpha.size), axis = 0)
    return np.concatenate([popsDot.flatten(),rdot.flatten(),ndot.flatten()])

def MoveAround(t,y,params):
    desire,neighborkey = params
    ysum = np.array([sum(y[ne]) for ne in neighborkey])
    desiresum = np.array([sum(desire[ne]) for ne in neighborkey])
    dot = desire*ysum - y*desiresum
    return dot

def TotChange(t,y,params):
    grwthParams,moveParams,speeds = params
    numpops = len(moveParams)
    dot = GrowthSys(t,y,grwthParams)
    for i in range(numpops):
        dot[i*len(moveParams[i][0]):(i+1)*len(moveParams[i][0])] = dot[i*len(moveParams[i][0]):(i+1)*len(moveParams[i][0])] + speeds[i]*MoveAround(t,y[i*len(moveParams[i][0]):(i+1)*len(moveParams[i][0])],moveParams[i])
    return dot

def GetNeighbors(l,size):
    if np.array(np.unravel_index(l,size))[1] == 0:
        return [np.ravel_multi_index((j,1),size) for j in range(size[1])]
    elif np.array(np.unravel_index(l,size))[1] == size[1]-1:
        return [np.ravel_multi_index((j,size[1]-1),size) for j in range(size[1])]
    else:
        return [np.ravel_multi_index(np.remainder(np.array(np.unravel_index(l,(100,100))) + [i,j],[size[0],size[1]+10]),size) for i in [-1,0,1] for j in [-1,0,1]]

class Demographics:
    def __init__(self,racenames,initialPop,baseKappas,baseGammas,basebeta1s,basebeta2s,mvspeeds):
        self.Races = racenames

        self.Populations = initialPop

        self.kappas = np.array([[[bk]*initialPop.shape[1]]*initialPop.shape[2] for bk in baseKappas]).flatten()
        self.gammas = np.array([[[bg]*initialPop.shape[1]]*initialPop.shape[2] for bg in baseGammas]).flatten()
        self.beta1s = np.array([[[bb1]*initialPop.shape[1]]*initialPop.shape[2] for bb1 in basebeta1s]).flatten()
        self.beta2s = np.array([[[bb2]*initialPop.shape[1]]*initialPop.shape[2] for bb2 in basebeta2s]).flatten()

        self.Chromosome = np.random.rand(len(racenames))

        self.History = dict([(racenames[i],[initialPop[i]]) for i in range(len(racenames))] + [("total",[np.sum(initialPop,axis = 0)])] + [("time",[0])])

        self.PoliticalHistory = [np.zeros_like(initialPop[0],dtype = int)]
        self.CountryNumber = 1

        self.ChangeFun = TotChange

        self.PreferredTemp = 0.75*np.ones(len(racenames))
        self.PreferredCrowd = np.ones(len(racenames))

        self.MovementSpeeds = mvspeeds
        self.CountrySizes = None
        self.ExistingCountries = None
        self.NumberCountries = 0

        self.Disasters = []
        self.Migrations = []
        self.Wars = {}

    def Desire(self,world):
        ResourceDesire =(world.Renewables + world.NonRenewables)/np.max(world.Renewables+ world.NonRenewables)
        TempDistSummer = np.array([dt - world.Temps[1] for dt in self.PreferredTemp])**2
        TempDistWinter = np.array([dt - world.Temps[0] for dt in self.PreferredTemp])**2
        TempDistSummer = 1 - TempDistSummer/np.max(TempDistSummer)
        TempDistWinter = 1 - TempDistWinter/np.max(TempDistWinter)
        TempDesire = (TempDistSummer + TempDistWinter)/2
        CrowdDist = np.array([prc - np.sum(self.Populations,axis = 0)/np.sum(self.Populations) for prc in self.PreferredCrowd])**2
        CrowdDesire = CrowdDist/np.max(CrowdDist)
        BasicDesire = (np.array([ResourceDesire]*len(self.Races)) + TempDesire + CrowdDesire)/3
        RiverAdj = BasicDesire*(0.1*np.array([world.RiverIndicator.reshape(world.GridSize).astype(float)]*len(self.Races)) + np.ones_like(BasicDesire))
        return (RiverAdj/np.max(RiverAdj))*np.array([world.LandIndicator.astype(float)]*len(self.Races))


    def ChangePop(self,world,dt = 0.1):
        grwall = ode(self.ChangeFun)
        grwall.set_initial_value(np.concatenate([self.Populations.flatten(),world.Renewables.flatten(),world.NonRenewables.flatten()]),0)
        desires = self.Desire(world)
        mvParams = [[des.flatten(),world.neighborkey] for des in desires]
        all_params = [[self.kappas,self.gammas,world.renewrate.flatten(),world.InitialRenew.flatten(),self.beta1s,self.beta2s,len(self.Races)],mvParams,self.MovementSpeeds]
        grwall.set_f_params(all_params)
        result = grwall.integrate(dt)
        newPops = result[:self.Populations.size].reshape(self.Populations.shape)
        newRenews = result[self.Populations.size:self.Populations.size + world.GlobeGrid[0].size].reshape(world.GridSize)
        newNon = result[self.Populations.size + world.GlobeGrid[0].size:].reshape(world.GridSize)
        for r in range(len(self.Races)):
            self.History[self.Races[r]] += [newPops[r,:,:]]
        self.History["total"] += [np.sum(newPops,axis = 0)]
        self.History["time"] += [self.History["time"][-1] + dt]
        self.Populations = newPops
        # if self.History["time"][-1]%1 < 0.01:
        NewPolitics,self.CountryNumber,Wars = self.GetPolitics(world)
        self.CountrySizes,self.ExistingCountries,self.NumberCountries = GetCountryTotals(NewPolitics)
        self.PoliticalHistory += [NewPolitics]
        self.Wars[self.History["time"][-1]] = Wars
        return newRenews,newNon


    def NaturalDisaster(self):
        whrind = np.random.randint(np.sum(self.History["total"][-1].astype(bool)))
        location = np.argwhere(self.History["total"][-1])[whrind]
        damage = 0.5*np.random.rand()
        for r in range(len(self.Races)):
            self.Populations[r,location[0],location[1]] = (1-damage)*self.Populations[r,location[0],location[1]]
        self.Disasters += [{"time":self.History["time"][-1],"location":location,"damage":damage}]

    def Migration(self,world):
        iii = np.random.choice(len(self.Races))
        whichRace = self.Races[iii]
        if np.sum(self.Populations[iii]):
            loc1 = np.random.choice(self.Populations[iii].size,p = self.Populations[iii].flatten().round(5)/np.sum(self.Populations[iii].flatten().round(5)))
            loc1 = np.unravel_index(loc1,self.Populations[iii].shape)
            loc1Spher = (world.GlobeGrid[0][loc1],world.GlobeGrid[1][loc1])
            loc1cart = toCart(*loc1Spher,1)
            time = self.History["time"][-1]
            mean_dist = min(np.pi*time/(0.001*world.Radius),np.pi/2)
            dist = min(np.random.exponential(mean_dist),np.pi)

            #a first vector on the circle
            if loc1Spher[0] + dist < np.pi:
                strt = (loc1Spher[0]+dist,loc1Spher[1])
            else:
                if loc1Spher[1] <0:
                    strt = (2*np.pi - loc1Spher[0]+dist,loc1Spher[1] + np.pi)
                else:
                    strt = (2*np.pi - loc1Spher[0]+dist,loc1Spher[1] - np.pi)

            strtC = toCart(*strt,1)

            #where on the circle:
            rot = 2*np.pi*np.random.rand()

            #rotate the start about the center of circle:
            rotation = R.from_rotvec(rot*np.array(loc1cart))
            finalCart = rotation.apply(strtC)

            finalSphere = toSphere(*finalCart)

            loc2 = (np.where(world.GlobeGrid[1][:,0] > finalSphere[2])[0][0],np.where(world.GlobeGrid[0][0] > finalSphere[1])[0][0])
            if world.LandIndicator[loc2]:
                prop_of_pop = 0.5*np.random.rand()
                size = self.Populations[iii][loc1]*(prop_of_pop)

                self.Populations[iii][loc1] = self.Populations[iii][loc1]-size
                self.Populations[iii][loc2] = size + self.Populations[iii][loc2]

                self.Migrations += [{"race":whichRace,"time":self.History["time"][-1],"start":loc1,"end":loc2,"size":size}]

    def GetPolitics(self,world):
        pop = np.sum(self.Populations, axis = 0)
        prev_pols = self.PoliticalHistory[-1]
        country_index = self.CountryNumber
        powers = np.random.exponential(SqrPowers(pop,world))
        new_countries,country_index = NewCountries(pop,prev_pols,country_index,world,powers)
        wars = []
        for ci in range(1,country_index+1):
            power = getPower(ci,new_countries,powers)
            if power > 0:
                num_squares = len(np.where(new_countries == ci)[0])
                ppsq = power/max(1,num_squares)
                if power > 1:
                    new_countries = ExpandCountry(ci,new_countries,world)
                if np.random.exponential(ppsq) > 2:
                    new_countries,victim = WhatIsItGoodFor(ci,new_countries,powers,world)
                    wars += [{"Aggressor":ci,"Defender":victim,"Revolution":False}]
                elif power < 0.01:
                    new_countries = DisbandCountry(ci,new_countries)
                elif ppsq<0.1 and num_squares>20:
                    new_countries,country_index = ViveLeRevolution(ci,new_countries,country_index,world)
                    wars += [{"Aggressor":country_index,"Defender":ci,"Revolution":True}]
        return new_countries,country_index,wars
