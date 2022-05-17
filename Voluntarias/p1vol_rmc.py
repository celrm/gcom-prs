from cvxpy import norm
import matplotlib.pyplot as plt
import numpy as np

def mid(p1,p2):
    return np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])

def norm(v1):
    return np.linalg.norm(v1)

def perp(v1):
    return np.array([-v1[1],v1[0]]) / norm(v1)

def tri(p1,p2):
    # third point of an equilateral triangle
    return mid(p1,p2) + perp(p2-p1)*norm(p2-p1)*np.sqrt(3)/2

""" Apartado i) """

max_level = 9

triangles = dict()
p01,p02 = np.array([0.,0.]),np.array([1.,0.])
p03 = tri(p01,p02)
triangles[0] = [[p01,p02,tri(p01,p02)]]
# plot new figure
fig = plt.figure()
plt.scatter([p01[0],p02[0],p03[0]],[p01[1],p02[1],p03[1]],color='k',s=0.1)
plt.savefig('img/triangle%d'%0)
plt.close()
points = dict()
points[0] = [p01,p02,p03]
for level in range(1,max_level):
    fig = plt.figure()
    triangles[level] = []
    points[level] = points[level-1]
    for [p1,p2,p3] in triangles[level-1]:
        tri_a = [p1,mid(p1,p2),mid(p1,p3)]
        tri_b = [mid(p1,p2),p2,mid(p2,p3)]
        tri_c = [mid(p1,p3),mid(p2,p3),p3]
        triangles[level] += [tri_a,tri_b,tri_c]
        # scatter points
        points[level] += [mid(p1,p2),mid(p1,p3),mid(p2,p3)]
        x,y = zip(*points[level])
        plt.scatter(x,y,color='k',s=0.1)

    plt.savefig('img/triangle%s'%str(level))
    plt.close()

    print(level,len(points[len(points)-1]))

""" Apartado ii) """

for level in range(2,max_level):
    ds = np.arange(0+(level+1)/max_level,1+(level+1)/max_level,0.1)
    fig = plt.figure()
    for d in ds:
        ms = []
        ns = range(1,100)
        for n in ns:    
            total = dict()
            vertical_stripes = {i:[] for i in range(n)}
            for l in range(level):
                for px, py in points[l]:
                    i = int(px*n)
                    if i == n:
                        i -= 1
                    vertical_stripes[i].append(py)

                for i in range(n):
                    for py in vertical_stripes[i]:
                        j = int(py*n)
                        total[i*n+j] = 1
                
                total_sum = len(total)
                ms.append(total_sum * (1/n) ** (2*d))
        # plot
        plt.plot(ns,ms,label=str(round(d,2)))
        if(ms[-1]<0.01):
            print(level,d)
            break
    plt.legend()
    plt.savefig('img/'+str(level)+'.png')
    plt.close()

print(np.log(3)/np.log(2))


