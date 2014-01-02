import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import matplotlib.animation as animation
#import douglasPeucker

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

    def buildParticleListAutoDP(self,tol,dt,x,lengthMin, minTime, compact=1):
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
            [psTemp, ixTemp] = dpSimplify(self.x[i],self.tol)
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

        if compact==1:
            self.compactToIndexVec()

    ############################################################ 

    def compactToIndexVec(self):
        print('Not complete')
        # Execute this script to compact x, etc..., prior to execution of 
        # self.continuousMSD, to eliminate all data not interesting (by 
        # determination of indexVec)

    def continuousMSD(self):
        self.MSD=[]
        self.MSDx=[]
        self.MSDy=[]
        self.avgMSD = []
        self.tau=[]
        self.t = []
        self.xs = []
        self.meanLogSlope=[]

        startIndex = np.int(np.round(self.maxTau/2/self.dt))
        meanRange = np.arange(np.int(np.round(self.slopeMin/self.dt)-1),np.int(np.round(self.slopeMax/self.dt)))
        for i in self.indexVec:
            lVecT = np.shape(self.x[i])[0]
            self.tau.append(np.arange(self.dt,(2*startIndex+1)*self.dt,self.dt))
            MSDxTemp = np.zeros([lVecT-2*startIndex,2*startIndex])
            MSDyTemp = np.zeros_like(MSDxTemp)
            MSDTemp = np.zeros_like(MSDxTemp)
            meanLogSlopeTemp = np.zeros(lVecT-2*startIndex)
           

            for j in range(startIndex,lVecT-startIndex):
                xVec = self.x[i][j-startIndex:j+startIndex+1,0]
                yVec = self.x[i][j-startIndex:j+startIndex+1,1]
                [MSDxTemp[j-startIndex,:], MSDyTemp[j-startIndex,:]] = MSDcalc(xVec,yVec,self.dt)

                if np.mod(j,200)==0:
                    print(j)

                A = np.vstack([np.log(self.tau[i][meanRange]),np.ones(len(meanRange))]).T
                #print(A.shape)
                m, c = np.linalg.lstsq(A,np.log(MSDxTemp[j-startIndex,meanRange]+MSDyTemp[j-startIndex,meanRange]))[0]
                #1/0
                #print(A)
                #print(MSDxTemp[meanRange])
                #print(m)
                meanLogSlopeTemp[j-startIndex] = m
                

            self.t.append(np.arange(0,MSDxTemp.shape[0]*self.dt,self.dt))
            self.MSDx.append(MSDxTemp)
            self.MSDy.append(MSDyTemp)
            self.MSD.append(MSDxTemp + MSDyTemp)
            self.meanLogSlope.append(meanLogSlopeTemp)
            self.avgMSD.append(np.mean((MSDxTemp + MSDyTemp),axis=0))
            self.xs.append(self.x[i][startIndex:lVecT-startIndex,:])

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # linFit script                          !
                #                                        !
                # Truncated xPos,t,meanLogSlope script   !
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # linFit = polyfit(log(tau{ii}(slopeMin/dt:slopeMax/dt))',log(MSD[ii}(slopeMin/dt:slopeMax/dt)),1)
                # meanLogSlope(ii) = linFit(1)

    ############################################################ 

    def plotMSD(self,index=0,DF=1):
        print('Displaying tMSD evolution')
        nFrames = np.int(np.floor(np.shape(self.MSD[index])[0]/DF))
        print(nFrames)

        fig = plt.figure(figsize=[6,4])
        ax = plt.axes()
        ax.set_xlim([np.min(self.tau[index]),np.max(self.tau[index])])
        ax.set_ylim([np.min(self.MSD[index]),np.max(self.MSD[index])])
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            y2 = self.MSD[index][DF*i,:]
            line.set_data(self.tau[index],y2)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=50, blit=True, repeat=False)
        plt.show()
        plt.close(fig)

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

################################################################################

def dpSimplify(x,tol):
    # Added 2013.12.28

    ps = [x[0,:]]
    ix = [0]
    i1=0
    i2=1

    while i2 < x.shape[0]:
        j = i1
        dComp=0
        while dComp < tol and j < i2:
            j+=1
            v1 = x[i2,:] - x[i1,:]
            v2 = x[j,:] - x[i1,:]
            vDiff = v2-np.dot(v1,v2)/np.dot(v1,v1)*v1
            #print('i1=',i1)
            #print('i2=',i2)
            #print('v1=',v1)
            #print('v2=',v2)
            #print('vDiff=',vDiff)
            dComp = np.sqrt(np.dot(vDiff,vDiff))
            #print('dComp=',dComp)
        
        #print('break')
        if dComp > tol:
            ix.append(i2-1)
            ps.append(x[i2-1,:])
            i1=i2-1
            #i2 = i1+1
            #print('ps=',ps)
        
        elif j == i2:
            i2 +=1
            #print('i2=',i2)
    #print('i2=',i2)
    
    ix.append(i2-1)
    ps.append(x[i2-1,:])
    
    return np.asarray(ps), np.asarray(ix)

