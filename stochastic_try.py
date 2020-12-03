jumpcount = 0

Scale = 1

gridpoints =  [(i,j) for i in range(World1.GlobeGrid[0].shape[0]) for j in range(World1.GlobeGrid[0].shape[1])]
RiverHere = [a in World1.RiverIndices for a in gridpoints]

InitialRenew = np.array([renew0(*p,World1) for p in gridpoints])
InitialRenew = np.floor(100*(InitialRenew - np.min(InitialRenew))/(np.max(InitialRenew)-np.min(InitialRenew)))/Scale
InitialNonRenew =np.array([initNonR(*p,World1) for p in gridpoints])
InitialNonRenew = np.floor(1000*(InitialNonRenew-np.min(InitialNonRenew))/(np.max(InitialNonRenew)-np.min(InitialNonRenew)))/Scale

Renews = [InitialRenew]
NonRenews = [InitialNonRenew]

PossibleStarts = np.where((World1.Elevation > World1.oLevel)*(World1.RainFall > 0.7))
pk = np.random.randint(len(PossibleStarts[0]))

thept = PossibleStarts[0][pk]*World1.GlobeGrid[0].shape[0] + PossibleStarts[1][pk]

InitialPop = np.zeros(len(gridpoints))
InitialPop[thept] = 100/Scale

population = [InitialPop]

AvgTemps = (World1.Temps[0] + World1.Temps[1])/2

growthrates = 0.01*np.ones(len(InitialPop))
deathrates = np.ones(len(InitialPop))
renewrate = np.ones(len(InitialRenew))

neighborar = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

t = 0
tarr = [t]

while (t < 10 and jumpcount < 1000):
    rmean =(sum(NonRenews[-1]) + sum(Renews[-1]))/(len(NonRenews[-1]) + len(Renews[-1]))
    desirabilities = [desire(population[-1][i],AvgTemps[gridpoints[i]],Renews[-1][i]+NonRenews[-1][i],rmean,100*World1.Radius*np.pi/World1.GlobeGrid[0].shape[0],0.75,RiverHere[i],Scale) for i in range(len(InitialPop))]

    ###Get our lambdas!
    gLams = lamGrowths(population[-1],Renews[-1],NonRenews[-1],growthrates)
    totalgLam =  np.sum(gLams)
    dLams = lamDeaths(population[-1],deathrates)
    totaldLam = np.sum(dLams)
    moveLams = lamLocalMovement(population[-1],World1,desirabilities,GCdist,mvmnt = 100)
    totalmoveLams = np.sum(moveLams)
    renewLams = lamRenews(Renews[-1],Renews[0],renewrate)
    totalRenew = np.sum(renewLams)

    lam0 = totalgLam + totaldLam + totalmoveLams + totalRenew

    ### time of next jump
    jmptime = np.random.exponential(1/lam0)
    t += jmptime
    tarr = tarr + [t]

    ###choose what the jump is
    u = np.random.rand()*lam0

    if u < totalgLam:
        #it's a birth! Yay!
        grarr = np.empty(len(gLams))
        grarr[0] = gLams[0]
        for k in range(1,len(gLams)):
            grarr[k] = gLams[k] + grarr[k-1]
        birth = np.where(u < grarr)[0][0]

        newpop = population[-1].copy()
        newpop[birth] = newpop[birth] + 1/Scale
        population += [newpop]

        newrenew = Renews[-1].copy()
        newrenew[birth] = newrenew[birth] - 1/Scale
        Renews += [newrenew]

        newnonrenews = NonRenews[-1].copy()
        newnonrenews[birth] = newnonrenews[birth] - 1/Scale
        NonRenews += [newnonrenews]

    elif u < totalgLam + totaldLam:
        #it's a death...bummer
        drarr = np.empty(len(dLams))
        drarr[0] = totalgLam
        for k in range(1,len(dLams)):
            drarr[k] = dLams[k] + drarr[k-1]
        death = np.where(u < drarr)[0][0]

        newpop = population[-1].copy()
        newpop[death] = newpop[death] - 1/Scale
        population += [newpop]

        NonRenews += [NonRenews[-1]]
        Renews += [Renews[-1]]

    elif u < totalgLam + totaldLam + totalmoveLams:
        #it's movement
        moveFrom = np.array([np.sum(lvlam) for lvlam in moveLams])
        fromarr = np.empty(len(moveFrom))
        fromarr[0] = totalgLam + totaldLam
        for k in range(1,len(moveLams)):
            fromarr[k] = moveFrom[k] + fromarr[k-1]
        leaving = np.where(u < fromarr)[0][0]

        toarr = np.empty(8)
        toarr[0] = fromarr[leaving]
        for k in range(1,8):
            toarr[k] = moveLams[leaving][k] + toarr[k-1]
        entering = np.where(u<toarr)[0][0]
        enteringIndex = World1.GlobeGrid[0].shape[0]*((gridpoints[leaving][0] + neighborar[entering][0]) % World1.GlobeGrid[0].shape[0]) + ((gridpoints[leaving][1] + neighborar[entering][1]) % World1.GlobeGrid[0].shape[1])

        newpop = population[-1].copy()
        newpop[leaving] = newpop[leaving] - 1/Scale
        newpop[enteringIndex] = newpop[enteringIndex] + 1/Scale
        population += [newpop]

        NonRenews += [NonRenews[-1]]
        Renews += [Renews[-1]]

    else:
        #resources returning
        retarr = np.empty(len(renewLams))
        retarr[0] = totalgLam + totaldLam + totalmoveLams
        for k in range(1,len(renewLams)):
            retarr[k] = renewLams[k] + retarr[k-1]
        renewing = np.where(u<retarr)[0][0]

        newRes = Renews[-1].copy()
        newRes[renewing] = newRes[renewing] + 1/Scale
        Renews += [newRes]

        population += [population[-1]]
        NonRenews += [NonRenews[-1]]


    jumpcount += 1



    
