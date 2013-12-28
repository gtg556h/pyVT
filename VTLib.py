import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import matplotlib.animation as animation
import douglasPeucker

class tmsd(object):

    def __init__(self, params, x):
        print('Initializing system')
        #self.x = [x,y]
        self.x = x
        self.tol = params['tol']
        self.theta0 = params['theta0']
        self.fps = params['fps']
        self.dt = params['dt']
        self.dx = params['dx']
        self.dy = params['dy']
        self.lengthMin = params['lengthMin']
        self.maxTau = params['maxTau']
        self.slopeMin = params['slopeMin']
        self.slopeMax = params['slopeMax']
        self.smootFactor = params['smoothFactor']
        self.cutoffRad = params['cutoffRad']
        self.nParticles = self.x.shape[0]
        self.minTime = params['minTime']
        #for i in range(0, self.nParticles)

        self.buildParticleListAutoDP(self.tol,self.dt,self.x,self.lengthMin, self.minTime)


    ######################################## 

    def buildParticleListAutoDP(self,tol,dt,x,lengthMin, minTime):
        # Automated selection of interesting particles, employing
        # Douglas Peucker algorithm for trajectory simplification,
        # 
        # Important parameters:
        # tol: Max displacement from smoothing line in Douglas-Peucker fit
        # lengthMin: 

        print('Building Particle List')
        
        self.ps=[]
        self.ix=[]

        indexVec = []
        migrateDist = []
        for i in range(0, self.nParticles):	
            [psTemp, ixTemp] = douglasPeucker.dpSimplify(self.x[i],self.tol)
            self.ps.append(psTemp)
            self.ix.append(ixTemp)
            nSegments = ixTemp.shape[0] - 1
            migrateDistTemp = 0
            for j in range(0,nSegments):
                migrateDistTemp += np.linalg.norm(psTemp[j+1,:]-psTemp[j,:])
            print(migrateDistTemp)
            migrateDist.append(migrateDistTemp)
            if migrateDistTemp > lengthMin and np.shape(x[i])[0] > minTime:
                indexVec.append(i)

        self.indexVec = indexVec
        self.migrateDist = migrateDist


    def continuousMSD(self):
        self.MSD=[]
        self.MSDx=[]
        self.MSDy=[]
        self.avgMSD = []
        self.tau=[]
        self.t = []
        self.xs = []
        meanLogSlope=[]

        startIndex = np.int(np.round(self.maxTau/2/self.dt))
        for i in self.indexVec:
            lVecT = np.shape(self.x[i])[0]

            MSDxTemp = np.zeros([lVecT-2*startIndex,2*startIndex])
            MSDyTemp = np.zeros_like(MSDxTemp)
            MSDTemp = np.zeros_like(MSDxTemp)

            for j in range(startIndex,lVecT-startIndex-1):
                xVec = self.x[i][j-startIndex:j+startIndex+1,0]
                yVec = self.x[i][j-startIndex:j+startIndex+1,1]
                [MSDxTemp[j-startIndex,:], MSDyTemp[j-startIndex,:]] = MSDcalc(xVec,yVec,self.dt)

                if np.mod(j,200)==0:
                    print(j)
                
            self.tau.append(np.arange(self.dt,(2*startIndex+1)*self.dt,self.dt))
            self.t.append(np.arange(0,MSDxTemp.shape[0]*self.dt,self.dt))
            self.MSDx.append(MSDxTemp)
            self.MSDy.append(MSDyTemp)
            self.MSD.append(MSDxTemp + MSDyTemp)
            self.avgMSD.append(np.mean((MSDxTemp + MSDyTemp),axis=0))
            self.xs.append(self.x[i][startIndex:lVecT-startIndex,:])

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # linFit script                          !
                #                                        !
                # Truncated xPos,t,meanLogSlope script   !
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # linFit = polyfit(log(tau{ii}(slopeMin/dt:slopeMax/dt))',log(MSD[ii}(slopeMin/dt:slopeMax/dt)),1)
                # meanLogSlope(ii) = linFit(1)


################################################################################
################################################################################

def MSDcalc(x,y,dt):
    lVecT = np.shape(x)[0]
    t = np.arange(0,lVecT*dt,dt)
    tau = np.arange(dt,lVecT*dt,dt)
    MSDx = np.zeros(lVecT-1)
    MSDy = np.zeros(lVecT-1)

    for i in range(1,lVecT):
        for j in np.arange(1,lVecT-i,i):
            MSDx[i] = MSDx[i] + ((x[i+j] - x[j])**2)/np.floor((lVecT-1)/i)
            MSDy[i] = MSDy[i] + ((y[i+j] - y[j])**2)/np.floor((lVecT-1)/i)

    return MSDx, MSDy

		
