#!/usr/bin/env python
# coding: utf-8

# In[14]:


#This code is rewritten from a Matlab code of the same name by Stephen Lucas (http://educ.jmu.edu/~lucassk/),
# it is developed as part of "Rolling Knots", a student project at the Heidelberg Experimental Geometry Lab.
#Link to the original paper: https://archive.bridgesmathart.org/2020/bridges2020-367.html

import numpy as np  #numpy
import scipy 
from mpl_toolkits import mplot3d
from scipy.spatial import ConvexHull
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

DPC = 10 #Decimal place Comparison
DPO = 2 #Decimal place output
MESH_LONG = 350
MESH_SMALL = 50


class Flatcircle:
    def __init__(self, centerpoint, radius, name = "Unnamed_2D_Circle"):
        self.name = name
        if isinstance(centerpoint, list):
            self.centerpoint = np.array(centerpoint) 
        if isinstance(centerpoint, np.ndarray): 
            self.centerpoint = centerpoint   
        self.radius = radius   
        self.x = self.centerpoint[0] 
        self.y = self.centerpoint[1]

    def __str__(self): 
        return "2D circle around ({}, {}) with radius {}.".format(round(self.centerpoint[0], DPO), round(self.centerpoint[1], DPO), round(self.radius, DPO))
    
    def __repr__(self): 
        return str(self)

    def __eq__(self, other): # Verify if the two circles are the same (up to the 10th decimals) - we don't want infinite intersection points. 
        return (round(np.linalg.norm(self.centerpoint - other.centerpoint), DPC) == 0) and round(self.radius, DPC) == round(other.radius, DPC)
    
    def __hash__(self): 
        return hash("({}, {}, {}, {})".format(self.p[0], self.p[1], self.radius))
    
    def intersect(self, other):

        r0=self.radius
        r1=other.radius
        x0=self.x
        y0=self.y
        x1=other.x
        y1=other.y

        
        d=scipy.spatial.distance.pdist([[x0, y0], [x1, y1]])
        X=[]

        if d==0 and r0 == r1:
            
            for j in np.linspace(0, 2 * np.pi, 800):
                X.append(np.array([r0*np.sin(j) + x0, r1*np.cos(j) + y0]))
        
        elif d <= r0 + r1 and d >= abs(r0 - r1):

            a=(r0**2-r1**2+d**2)/(2*d)
            h=np.sqrt(r0**2-a**2)
            x2=x0+a*(x1-x0)/d   
            y2=y0+a*(y1-y0)/d   
            x3=x2+h*(y1-y0)/d     
            y3=y2-h*(x1-x0)/d 

            x4=x2-h*(y1-y0)/d
            y4=y2+h*(x1-x0)/d
            
            X=[np.array([x3[0], y3[0]]), np.array([x4[0], y4[0]])]
            #print("intersect", len(X[0]-np.array([x0, y0])), len(np.array([x0, y0])), r0, r1)
            #print (X, "XXXXXXX")
        

        return X

class Morton_Knot: #Parametrization from Morton's paper on tritangentless knots. 
    def __init__(self, a = 0.5, z_scale = 1, sgpp = 1, name = "Unnamed_2D_Circle"):
        self.name = name
        self.a = a
        t = np.linspace(0, 2 * np.pi, MESH_LONG, endpoint=False)
        self.p = 3
        self.q = 2
        self.scale = z_scale
        self.sgpp = sgpp   #Stereographic projection parameter
        self.b = np.sqrt(1 - a ** 2)
        c = self.a / (1 + self.b) 
        R = 1 / (1 + self.b)
        r = self.b / (1 + self.b)
        denom = self.sgpp - self.b * np.sin(self.q * t)
        self.x = c * self.a * np.cos(self.p * t) / denom
        self.y = c * self.a * np.sin(self.p * t) / denom
        self.z = self.scale * c * self.b * np.cos(self.q * t) / denom
        points = []
        for i in range(len(self.x)):
            points.append(np.array([self.x[i], self.y[i], self.z[i]]))
        self.points = points

    def __str__(self):
        return "Mortons knot with param a = {}.".format(round(self.a, DPO))
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (round(np.linalg.norm(self.a - other.a), DPC) == 0)
    
    def __hash__(self):
        return hash("Mortons knot with param a = {}.".format(self.a))

    def plot(self):
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        c = self.x + self.y
        ax.scatter(self.x, self.y, self.z, c = c)
        ax.set_title('3d Scatter plot')
        plt.show()


class Rolling_Knot:
    def __init__(self, knot, name = "Unnamed_Rolling_Knot"):
        self.name = name
        self.knot = knot
        self.hull = ConvexHull(self.knot.points) 

        self.hull_count = 0
        for triangle_pos in self.hull.simplices:
            self.hull_count += 1
        
        self.sorthull() #TODO     
        self.calc_planar_triangles() #TODO
        self.align_rolling() #TODO

    def __str__(self):
        return "Rolling Knot: " + str(self.knot)
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.knot == other.knot
    
    def __hash__(self):
        return hash("Rolling Knot: " + str(self.knot))

    def sorthull(self):

        for j in range(len(self.hull.simplices)):
        
            # makes 1-2, 2-3 long and 3-1 short
            i0 = self.hull.simplices[j][0]
            i1 = self.hull.simplices[j][1]
            i2 = self.hull.simplices[j][2]
            d01=np.linalg.norm(self.hull.points[i0]-self.hull.points[i1]) #len1
            d02=np.linalg.norm(self.hull.points[i0]-self.hull.points[i2]) #len3
            d12=np.linalg.norm(self.hull.points[i1]-self.hull.points[i2]) #len2
            
            if d01<=d12 and d01<=d02: 
                [self.hull.simplices[j][0],self.hull.simplices[j][1],self.hull.simplices[j][2]]=[i1, i2, i0]
            elif d12<=d02: 
                [self.hull.simplices[j][0],self.hull.simplices[j][1],self.hull.simplices[j][2]]=[i2, i0, i1]
           
            # repairs orientation
            i0 = self.hull.simplices[j][0]
            i1 = self.hull.simplices[j][1]
            i2 = self.hull.simplices[j][2]

            if np.dot(np.cross(self.hull.points[i0]-self.hull.points[i1], self.hull.points[i2]-self.hull.points[i1]), self.hull.points[i1]) > 0:
                self.hull.simplices[j][0]=i2
                self.hull.simplices[j][2]=i0   

    def calc_planar_triangles(self):
        # trixy holds the triangle locations mapped to the plane
        trixy = np.zeros([2 * self.hull_count,6])
        # origin holds the location of the center of mass when a particular
        # triangle is on the surface
        origin = np.zeros([2 * self.hull_count,3])

        # First triangle, put side 1-2 on the x-axis
        a = self.knot.points[self.hull.simplices[0][0]]
        b = self.knot.points[self.hull.simplices[0][1]]
        c = self.knot.points[self.hull.simplices[0][2]]
        len1 = np.linalg.norm(a-b)
        len2 = np.linalg.norm(b-c)
        len3 = np.linalg.norm(c-a)
        p = [0,0] 
        q = [len1,0]

        # Identify potential locations for 3rd point
        c_1 = Flatcircle(p,len3)
        c_2 = Flatcircle(q,len2)
        [r_1, r_2] = c_1.intersect(c_2)
        if turn(p,q,r_1) > 0:  # Choose left hand turn
            r = r_1
        else:
            r = r_2
        trixy[0]=[p[0], p[1], q[0], q[1], r[0], r[1]]

        type1 = 1 # Triangle has long edge first 
        # Which edge to search for next
        next = [self.hull.simplices[0][2], self.hull.simplices[0][1]]

        # Identify location of the center of mass
        locm = sphereint(p[0], p[1], np.linalg.norm(a), q[0], q[1], np.linalg.norm(b), r[0], r[1], np.linalg.norm(c))
        
        if locm[0][2] > 0:
            origin[0]=locm[0]
        else:
            origin[0]=locm[1]
        
        for i in range(1, 2*self.hull_count): # For the rest of the triangles
            # Identify which one has next as a long pair
            j = 0
            done = False
            type2 = 1
            while False == done:
                if self.hull.simplices[j][0] == next[0] and self.hull.simplices[j][1] == next[1]:
                    done = True
                   
                elif self.hull.simplices[j][1] == next[0] and self.hull.simplices[j][2] == next[1]:
                    done = True
                    type2 = 2 
                else:
                    j += 1

            # Identify next triangle lengths
            a = self.knot.points[self.hull.simplices[j][0]]
            b = self.knot.points[self.hull.simplices[j][1]]
            c = self.knot.points[self.hull.simplices[j][2]]
            len1 = np.linalg.norm(a-b)
            len2 = np.linalg.norm(b-c)
            len3 = np.linalg.norm(c-a)
             
            # Identify the other long edge that will be connected to next, and what to connect to
            if type1 == 1:
                if type2 == 1:
                    p = r
                    c_1 = Flatcircle(p,len3)
                    c_2 = Flatcircle(q,len2)
                    [r_1, r_2] = c_1.intersect(c_2)
                    if turn(p,q,r_1) > 0:  
                        r = r_1
                    else:
                        r = r_2
                else:
                    [r, q] = [q, r]
                    c_1 = Flatcircle(q,len1)
                    c_2 = Flatcircle(r,len3)
                    [p_1, p_2] = c_1.intersect(c_2)
                    if turn(p_1,q,r) > 0:  
                        p = p_1
                    else:
                        p = p_2
            else:
                if type2 == 1:
                    [p, q] = [q, p]
                    c_1 = Flatcircle(p,len3)
                    c_2 = Flatcircle(q,len2)
                    [r_1, r_2] = c_1.intersect(c_2)
                    if turn(p,q,r_1) > 0:  
                        r = r_1
                    else:
                        r = r_2
                else:
                    r = p
                    c_1 = Flatcircle(q,len1)
                    c_2 = Flatcircle(r,len3)
                    [p_1, p_2] = c_1.intersect(c_2)
                    if turn(p_1,q,r) > 0:  
                        p = p_1
                    else:
                        p = p_2

            # Update next side to connect to
            if type2 == 1:
                next = [self.hull.simplices[j][2], self.hull.simplices[j][1]]
            else:
                next = [self.hull.simplices[j][1], self.hull.simplices[j][0]]

            type1=type2 # Update kind of triangle
            trixy[i]=[p[0], p[1], q[0], q[1], r[0], r[1]] # Place triangle
            
            locm = sphereint(p[0], p[1], np.linalg.norm(a), q[0], q[1], np.linalg.norm(b), r[0], r[1], np.linalg.norm(c))
            
            if locm[0][2] > 0:
                origin[i]=locm[0]
            else:
                origin[i]=locm[1]
    
            
        self.planar_triangles = trixy
        self.center_of_mass = origin

    def align_rolling(self):
        
        trixy = self.planar_triangles
        origin = self.center_of_mass
        
        # To rotate so moving up the page, identify last triangle edge that is horizontal, and find the left point
        
        ltr = trixy[2*self.hull_count-1]
        p=[ltr[0], ltr[1]]
        q=[ltr[2], ltr[3]]
        r=[ltr[4], ltr[5]]
        lp = [0,0]
        
        if abs(p[1] - q[1]) < abs(q[1] - r[1]):
            if p[0] < q[0]:
                lp = p
            else:
                lp = q
        else:
            if q[0] < r[0]:
                lp = q
            else:
                lp = r
        
        
        #Identify angle to rotate so that (lastx,lasty) is on positive x axis.
        
        w = np.arctan(lp[0] / lp[1])
        
        if lp[1]<0:
            w += np.pi
            
        # Now rotate every point
    
        for j in range(2*self.hull_count):
            [pj1,pj2] = [np.cos(w)*trixy[j][0]-np.sin(w)*trixy[j][1], np.sin(w)*trixy[j][0] + np.cos(w)*trixy[j][1]]
            [qj1,qj2] = [np.cos(w)*trixy[j][2]-np.sin(w)*trixy[j][3], np.sin(w)*trixy[j][2] + np.cos(w)*trixy[j][3]]
            [rj1,rj2] = [np.cos(w)*trixy[j][4]-np.sin(w)*trixy[j][5], np.sin(w)*trixy[j][4] + np.cos(w)*trixy[j][5]]
            trixy[j] = [pj1, pj2, qj1, qj2, rj1, rj2]
            
        # And rotate the origins
        
        for j in range(2*self.hull_count):
            [xj, yj] = [np.cos(w)*origin[j][0]-np.sin(w)*origin[j][1], np.sin(w)*origin[j][0] + np.cos(w)*origin[j][1]]
            origin[j][0] = xj
            origin[j][1] = yj
        
    
     # Shift bottom left to origin 
    
        self.planar_triangles = trixy
        self.center_of_mass = origin


    def plot(self):
        fig, ax = plt.subplots(ncols=2, figsize=(16, 9))
        ax[0].set_title('Rolling pattern')
        ax[0].set_aspect('equal')
        for triangle in self.planar_triangles:
            ax[0].plot([triangle[0],triangle[2],triangle[4],triangle[0]], [triangle[1],triangle[3],triangle[5],triangle[1]], 'k-', linewidth=0.5)
        ax[0].plot(self.center_of_mass[:,0], self.center_of_mass[:,1], color='r')
        ax[1].set_title('Mass Deviation')
        ax[1].plot(self.center_of_mass[:,1], self.center_of_mass[:,2] - sum(self.center_of_mass[:,2])/len(self.center_of_mass[:,2]), color='r', marker = '.', markerfacecolor = 'r', label ='Height')
        ax[1].plot(self.center_of_mass[:,1], self.center_of_mass[:,0] - sum(self.center_of_mass[:,0])/len(self.center_of_mass[:,0]), color='b', marker = 's', markerfacecolor = 'b', label ='Transversal')
        ax[1].legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)
        plt.show()
        

def turn(p,q,r):
    # Returns   positive if p-q-r makes a left hand turn in the plane,
    #           negative if a right hand turn and 
    #           zero if collinear  (x2−x1)(y3−y1)−(y2−y1)(x3−x1)
    return ((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))
    
    
def sphereint(x0,y0,r0,x1,y1,r1,x2,y2,r2):
    # Function to locate the intersection of the spheres 
    # The following implementaton is based on Wikipedia Trilateration article. 
    P0 = np.array([x0, y0, 0])
    P1 = np.array([x1, y1, 0])
    P2 = np.array([x2, y2, 0])
    temp0 = P1-P0                                      
    e_x = temp0/np.linalg.norm(temp0)                              
    temp1 = P2-P0                                        
    i = np.dot(e_x,temp1)                                   
    temp2 = temp1 - i*e_x                                
    e_y = temp2/np.linalg.norm(temp2)                           
    e_z = np.cross(e_x,e_y)                                 
    d = np.linalg.norm(P1-P0)                                      
    j = np.dot(e_y,temp1)                                   
    x = (r0*r0 - r1*r1 + d*d) / (2*d)                    
    y = (r0*r0 - r2*r2 -2*i*x + i*i + j*j) / (2*j)       
    temp3 = r0*r0 - x*x - y*y                            
    if temp3 < 0:                                          
        return "The three spheres do not intersect."
    z = sqrt(temp3)                                      
    p_12_a = P0 + x*e_x + y*e_y + z*e_z                  
    p_12_b = P0 + x*e_x + y*e_y - z*e_z                  
    return [p_12_a, p_12_b]

def objective_function(parameters):
    #Knot to test for rolling ability
    test_knot = Rolling_Knot(Morton_Knot(a = parameters[0], z_scale = parameters[1], sgpp = parameters[2]))
    ltr = test_knot.planar_triangles[2*test_knot.hull_count-1]
    norm = max([ltr[1], ltr[3], ltr[5]])

    T = len(test_knot.center_of_mass[:,0])
    v_trans = np.zeros((T-1,3))
    vx_2 = np.zeros(T-1)
    basic_cand = 0.0
    sum_v_trans = sum(v_trans[:][0])
    for t in range(T-1):
        v_trans[t,:] = test_knot.center_of_mass[t+1,:]-test_knot.center_of_mass[t,:] # translational
        # velocity of center of mass
    basic_cand += (max(v_trans[:][0]) - min(v_trans[:][0]))

    return basic_cand/norm

def objective_function_2(parameters):
    #Knot to test for rolling ability
    test_knot = Rolling_Knot(Morton_Knot(a = parameters[0], z_scale = parameters[1], sgpp = parameters[2]))
    ltr = test_knot.planar_triangles[2*test_knot.hull_count-1]
    norm = max([ltr[1], ltr[3], ltr[5]])

    d_y = max(test_knot.center_of_mass[:,2]) - min(test_knot.center_of_mass[:,2])
    d_z = max(test_knot.center_of_mass[:,0]) - min(test_knot.center_of_mass[:,0])

    #Basic candidate -- We add the maximal lateral and vertical deviations together 
    basic_cand = np.abs(d_y) + np.abs(d_z*4)
    
    return basic_cand/norm

def optimize_knot(_a_min = 0.96, _a_max = 0.99, _z_scale_min = 2.495, _z_scale_max = 4.0, _sgpp_min = 0.4, _sgpp_max = 0.9):
    #https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/

    #Base parameter values are given either by definition (a) or practicality (z).
    #Running the optimization and consequently adjusting the boundaries is a valid method 
    #to finding better solutions.

    a_min = _a_min
    a_max = _a_max
    z_scale_min = _z_scale_min
    z_scale_max = _z_scale_max 
    sgpp_min = _sgpp_min
    sgpp_max = _sgpp_max
    s_a = a_min + random.random() * (a_max - a_min)
    s_z_scale = z_scale_min + random.random() * (z_scale_max - z_scale_min)
    s_sgpp = sgpp_min + random.random() * (sgpp_max - sgpp_min)
    
    #Staring point. We select a (semi) random point withing the parameter boundaries.
    pt = [s_a, s_z_scale, s_sgpp]
    
    #Search for a result
    result = scipy.optimize.minimize(objective_function, pt, method='nelder-mead') 
    
    #Summarize the result
    #print('Status : %s' % result['message'])
    print(result['x'])

def plot_objective():
    ra_min = 0.05
    ra_max = 0.50
    rz_min = 0.51   #To plot a larger value range the N_MESH value needs to be adjusted. A greater mesh size
    rz_max = 0.99   #is needed for a larger value range.
    L = 60           

    a = np.linspace(ra_min, ra_max, L)
    z = np.linspace(rz_min, rz_max, L)
    av, zv = np.meshgrid(a, z)
    
    results = np.zeros((L, L))

    for i in range(0, L):
        for j in range(0, L):
            x = [a[i], z[j]]
            results[i][j] = objective_function(x)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    surf = ax.plot_surface(av, zv, results, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    # Customize the z axis.
    ax.set_zlim(0.0, 0.0701)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel("a")
    ax.set_ylabel("z")
    ax.set_zlabel("Objective Function")
    plt.show()

def plot_objective_4d():
    ra_min = 0.01
    ra_max = 0.51
    rz_min = 0.01   
    rz_max = 0.51   
    rsgpp_min = 1.1     #It's important that sgpp > b, otherwise the knot will not be well-defined
    rsgpp_max = 2.1
    L = 10

    a = np.linspace(ra_min, ra_max, L)
    z = np.linspace(rz_min, rz_max, L)
    sgpp = np.linspace(rsgpp_min, rsgpp_max, L)

    av, zv, sgppv = np.meshgrid(a, z, sgpp)
   
    results = np.zeros((L, L, L))

    for i in range(0, L):
        for j in range(0, L):
            for k in range(0, L):
                x = [a[i], z[j], sgpp[k]]
                results[i][j][k] = objective_function(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(av, zv, sgppv, c = results, cmap = cm.coolwarm, vmin = 0, vmax = 0.007)
    cbar = fig.colorbar(img)
    cbar.set_label("Objective function")
    ax.set_xlabel('a')
    ax.set_ylabel('z')
    ax.set_zlabel('sgpp')

    plt.show()

def main():
    knot = Morton_Knot(0.45613167, 0.4631802,  1.40826749)
    rolling_knot = Rolling_Knot(knot)
    #print(rolling_knot.planar_triangles)
    rolling_knot.plot()
    #plot_objective_4d()
    #for i in range(10):
    #    optimize_knot()
    
    
    

if __name__ == "__main__": #This code is executed if this file is in the main namespace
    #####Time optimization#####
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats()


# In[ ]:





# In[ ]:




