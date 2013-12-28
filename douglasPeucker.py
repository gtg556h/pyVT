import numpy as np

# Call:
# [ps, ix] = douglasPeucker.dpsimplify(x,tol)


def dpSimplify(x,tol):
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

