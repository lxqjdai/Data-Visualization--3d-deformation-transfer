#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
data = scipy.io.loadmat('./vs.mat') 
vertices = data['vertices']
triangles = data['triangle']


# In[2]:


#vertices = vertices['target_vertices']


# In[3]:


#triangles = triangles['triangles']


# In[4]:


obj_file = './result.obj'
with open(obj_file, 'w') as f:
    for v in range(0,vertices.shape[0]):
        f.write('v %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (vertices[v,0],vertices[v,1],vertices[v,2],0,0,0))

    for t in range(0,triangles.shape[0]):
        f.write('f {} {} {}\n'.format(*triangles[t,:]+1))

print('Calculated the isosurface, save at obj file:',obj_file)

