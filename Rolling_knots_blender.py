import bpy
import bmesh
import numpy as np
import pip
#pip.main(['install', 'scipy'])
from scipy.spatial import ConvexHull


DPC = 10 #Decimal place Comparison
DPO = 2 #Decimal place output
MESH_LONG = 800
MESH_SMALL = 50 
                                         

class Curve:
    def __init__(self, points, small_radius = 1, name = "Unnamed_Curve"):
        self.name = name
        self.r = small_radius
        self.points = []
        for p in points:
            if isinstance(points, list):
                self.points.append(np.array(p))
            if isinstance(points, np.ndarray):
                self.points.append(p)
        self.sample_size = len(self.points)
         
    def __str__(self):
        return "Curve: {}, with line radius {}. Sampled at {} points".format(
                    self.name, str(self.r), str(len(self.points)))
        
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        point_compare = 0 
        
        if (self.sample_size != other.sample_size):
            return False
        else:
            for i in range(0, len(self.points)):
                if (round(np.linalg.norm(self.points[i] - other.points[i]), DPC) != 0):
                    point_compare = 1 
                   
                return ((round(self.r - other.r, DPC) == 0) and (point_compare == 0))
    
    def plot(self):
        
        vertices = []
        edges = []                      
        faces = []
        
        n_3 = np.array([0,0,1])
        n_1 = np.array([1,0,0])
        n_2 = np.array([0,1,0])
        
        vert_0=[]
        diff=[]
        projec=[]
        theta = 2 * np.pi / MESH_SMALL
        l=len(self.points)
        t=int(l/MESH_LONG)
        eps=0.001
        eps_2=1
        x_0=np.array([0,0,0])
        
        #Points
        for i in range(0, MESH_LONG):
            vert_0.append(self.points[i*t])
            
          
        vert_0.append(self.points[t])
            
        for i in range(0, MESH_LONG):
            basedir = vert_0[i]
            diff=(vert_0[i+1] - vert_0[i])
            if basedir[0] < eps:
                projec.append(basedir[1]*n_2 +  basedir[2]* n_3)
            else:
                if diff[0] > eps:
                    projec.append((diff[1]*n_2 +  diff[2]* n_3)* (basedir[0]/diff[0]))
                    
        j=0
        
        while j < len(projec):
            if np.linalg.norm(projec[j]-x_0) > eps_2:
                j += 1
            else:
                x_0 = x_0 + 10*np.random.random_sample()*n_2 + 10*np.random.random_sample()*n_3
                j=0
            
        for i in range(0, MESH_LONG):
            basedir = vert_0[i]
            diff = vert_0[i+1] - vert_0[i]
            diff = diff/np.linalg.norm(diff)
            
        
            w_0=(basedir - x_0)
            w_0=w_0/np.linalg.norm(w_0)
            #Why use the dot product? 
            w_2=w_0-(np.dot(diff, w_0)*diff)
            w_2=w_2/np.linalg.norm(w_2)
            w_1=np.cross(w_2, diff)
            for j in range(0, MESH_SMALL):
                vertices.append(basedir + w_1 * self.r * (-1) * np.cos(j * theta)
                                        + w_2 * self.r * np.sin(j * theta))
                                         
        #Faces 
        for i in range(0, MESH_LONG):
            for j in range(0, MESH_SMALL):
                faces.append((MESH_SMALL * i + j, 
                              MESH_SMALL * ((i + 1) % MESH_LONG) + j,
                              MESH_SMALL * ((i + 1) % MESH_LONG) + ((j + 1) % MESH_SMALL), 
                              MESH_SMALL * i + ((j + 1) % MESH_SMALL)))         
        
        new_mesh = bpy.data.meshes.new('new_mesh')
        new_mesh.from_pydata(vertices, edges, faces) #Auto generates edges with faces
        new_mesh.update()            
    
        new_object = bpy.data.objects.new("Objekt", new_mesh)
        
        bpy.context.scene.collection.objects.link(new_object) 
        
class Morton_Knot:
    def __init__(self, a = 1/2, z_scale = 1, sgpp = 1, scaling_factor = 10, small_radius = 1, name = "Unnamed_Morton_Knot"):
        
        self.b = (1-a**2)**0.5
        self.a = a
        self.c = scaling_factor
        self.r = small_radius
        self.z_scale = z_scale
        self.sgpp = sgpp
        self.name = name
         
    def __str__(self):
        return "Morton knot: {}, with line radius {}, parameter a = {}, and scaling constant c = {}.".format(
                    self.name, str(self.r), str(self.a), str(self.c))
        
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return (round(self.a - other.a, DPC) == 0 and round(self.r - other.r, DPC) == 0 and
                round(self.c - other.c, DPC) == 0)
  
    def x(self, t):
        return self.c*self.a*np.cos(3*t)/(self.sgpp-self.b*np.sin(2*t))
        
    def y(self, t):
        return self.c*self.a*np.sin(3*t)/(self.sgpp-self.b*np.sin(2*t))
        
    def z(self, t):
        return self.z_scale*self.c*self.b*np.cos(2*t)/(self.sgpp-self.b*np.sin(2*t))
        
    def plot(self):
        vertices = []
        edges = []                      
        faces = []
        
        theta = 2 * np.pi / MESH_SMALL
        phi = 2 * np.pi / MESH_LONG
        
        n_3 = np.array([0,0,1])
        n_1 = np.array([1,0,0])
        n_2 = np.array([0,1,0])
        vert_0 = []
        
        #Vertices
        for i in range(0, MESH_LONG+1):
            t=i*phi
            vert_0.append(self.x(t) * n_1 + self.y(t) * n_2 + self.z(t) * n_3)
        
        for i in range(0, MESH_LONG):
            basedir = vert_0[i]
            diff = vert_0[i+1] - vert_0[i]
            diff = diff/np.linalg.norm(diff)
            w_0=basedir/np.linalg.norm(basedir) 
            w_2=w_0-(np.dot(diff, w_0)*diff)
            w_2=w_2/np.linalg.norm(w_2)
            w_1=np.cross(w_2, diff)
            
            for j in range(0, MESH_SMALL):
                vertices.append(basedir + w_1  * self.r * (-1) * np.cos(j * theta)
                                        + w_2  * self.r * np.sin(j * theta))
                                         
        #Faces 
        for i in range(0, MESH_LONG):
            for j in range(0, MESH_SMALL):
                faces.append((MESH_SMALL * i + j, 
                              MESH_SMALL * ((i + 1) % MESH_LONG) + j,
                              MESH_SMALL * ((i + 1) % MESH_LONG) + ((j + 1) % MESH_SMALL), 
                              MESH_SMALL * i + ((j + 1) % MESH_SMALL)))         
        
        new_mesh = bpy.data.meshes.new('new_mesh')
        new_mesh.from_pydata(vertices, edges, faces) #Auto generates edges with faces
        new_mesh.update()            
    
        new_object = bpy.data.objects.new("Objekt", new_mesh)
        
        bpy.context.scene.collection.objects.link(new_object)    
    
class Torus:
    def __init__(self, direction, big_radius = 2, small_radius = 1, name = "Unnamed_Torus"):
        self.name = name
        self.r = small_radius
        self.R = big_radius
        if isinstance(direction, list):
            self.direction = np.array(direction)
        if isinstance(direction, np.ndarray):
            self.direction = direction
        self.normal = self.direction / np.linalg.norm(self.direction)
            
    def __str__(self):
        return "Torus: {}, with big radius {}, small radius {} and hole pointing in diretion ({}, {}, {}).".format(
                    self.name, str(self.R), str(self.r), str(round(self.normal[0], DPO)), str(round(self.normal[1], DPO)), 
                    str(round(self.normal[2], DPO)))
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return (round(self.r - other.r, DPC) == 0) and (round(self.R - other.R, DPC) == 0) and (round(
                    np.linalg.norm(self.p - other.p), DPC) == 0)
        
    def plot():
        vertices = []
        edges = []                      
        faces = []
        
        theta = 2 * np.pi / MESH_SMALL
        phi = 2 * np.pi / MESH_LONG
        
        n_v = self.normal
        random_direction = np.array([0, 0, 1])
        if np.linalg.norm(n_v - random_direction) < 0.1: 
            random_direction = np.array([0, 1, 0]) 
                
        v_1 = np.cross(random_direction, n_v)
        v_2 = np.cross(n_v, v_1)
        
        n_1 = v_1 / np.linalg.norm(v_1)
        n_2 = v_2 / np.linalg.norm(v_2)
        
        #Points
        for i in range(0, MESH_LONG):
            basedir = np.cos(i*phi) * n_1 + np.sin(i*phi) * n_2
            basedir = basedir / np.linalg.norm(basedir)
            for j in range(0, MESH_SMALL):
                temp = basedir * self.R 
                vertices.append(temp + basedir * self.r * (-1) * np.cos(j * theta)
                                         + n_v * self.r * np.sin(j * theta))
        #Faces 
        for i in range(0, MESH_LONG):
            for j in range(0, MESH_SMALL):
                faces.append((MESH_SMALL * i + j, 
                              MESH_SMALL * ((i + 1) % MESH_LONG) + j,
                              MESH_SMALL * ((i + 1) % MESH_LONG) + ((j + 1) % MESH_SMALL), 
                              MESH_SMALL * i + ((j + 1) % MESH_SMALL)))         
        
        new_mesh = bpy.data.meshes.new('new_mesh')
        new_mesh.from_pydata(vertices, edges, faces) #Auto generates edges with faces
        new_mesh.update()            
    
        new_object = bpy.data.objects.new(self.name, new_mesh)
        
        bpy.context.scene.collection.objects.link(new_object) 
        
        
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

    def __eq__(self, other):
        return (round(np.linalg.norm(self.centerpoint - other.centerpoint), DPC) == 0) and round(self.radius, DPC) == round(other.radius, DPC)
    
    def __hash__(self):
        return hash("({}, {}, {}, {})".format(self.p[0], self.p[1], self.radius)) 
        
def delete_screen():
    coll = bpy.context.scene.collection
    if coll:
       for obj in coll.objects:
           obj.hide_set(False)
    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete(use_global = False, confirm = False)
    

def main():
    delete_screen()
    
    knot = Morton_Knot(a = 0.1473, z_scale = 0.3592, sgpp = 1.0, scaling_factor = 5, small_radius = 0.3, name = "Tollert Knoten")

    #---Morton_Knot.plot() test
    knot.plot()
    
if __name__ == "__main__": #This code is executed if this file is in the main namespace
    #####Time optimization#####
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats()