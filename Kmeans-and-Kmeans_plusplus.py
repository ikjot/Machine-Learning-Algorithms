
# coding: utf-8

# In[4]:


import numpy as np
import scipy.io as spio
import random as rn
import matplotlib.pyplot as plt
mat = spio.loadmat('kmeans_data.mat')
data = mat['data']


# In[5]:


def calculate_distance(new_centres, point):
    #calculating the distance between points
    diff = new_centres - point
    distance = np.sum(np.square(diff),axis = 1)
    return distance


# In[9]:


# K-means
new_centres = []
loss_function = []

for no_of_clusters in range(2,11) :
    matrix = np.zeros(data.shape[0]) 
    random_choice = np.random.choice(range(data.shape[0]),size = no_of_clusters, replace = False)
    new_centres = data[random_choice]
    prev_centres = new_centres * 0.1
    while abs(np.max(prev_centres - new_centres)) >= 0.01 :
        for j in range(data.shape[0]) :
            distances = calculate_distance(new_centres,data[j])
            matrix[j] = np.argmin(distances)
        prev_centres = new_centres.copy()
        for cluster in range(no_of_clusters) :
            new_centres[cluster] = np.mean(data[np.where(matrix == cluster)],axis = 0)
    loss = 0
    for i in range(no_of_clusters):
        distances_temp = calculate_distance(data[np.where(matrix == i)],new_centres[i])
        square_root = np.sqrt(distances_temp)
        loss += np.sum(square_root)
    loss_function.append(loss)
print(loss_function)  
    


# In[10]:


x_temp = range(2,11)
plt.plot(x_temp,loss_function)
plt.show()


# In[11]:


# K-means++
new_centres = []
loss_K_plus_plus_function = []

for no_of_clusters in range(2,11) :
    dist = []
    for i in range(data.shape[0]):
        d = calculate_distance(data,data[i])
        dist.append(d)
    dist = np.array(dist)
    (p1,p2) = ( int(np.argmax(dist) / dist.shape[1]) , int(np.argmax(dist) % dist.shape[1]) )
    selected = [p1,p2]
    new_centres = [data[p1],data[p2]]
    for i in range(2,no_of_clusters):
        dist = []
        for j in range(data.shape[0]):
            if j not in selected:
                distances_temp = calculate_distance(np.array(new_centres),data[j])
                sum_temp = np.sum(distances_temp)
                dist.append(sum_temp)
            else:
                dist.append(-1)
        dist = np.array(dist)
        p = np.argmax(dist)
        selected.append(p)
        new_centres.append(data[p])
    new_centres = np.array(new_centres)
    
    #matrix = np.zeros(data.shape[1]) 

    matrix = np.zeros(data.shape[0])    
    prev_centres = new_centres * 0.1
    
    while abs(np.max(prev_centres - new_centres)) >= 0.01 :
        for j in range(data.shape[0]) :
            d_temp = calculate_distance(new_centres,data[j])
            matrix[j] = np.argmin(d_temp)
        prev_centres = new_centres.copy()
        for cluster in range(no_of_clusters) :
            new_centres[cluster] = np.mean(data[np.where(matrix == cluster)],axis = 0)
    loss = 0
    for i in range(no_of_clusters):
        loss += np.sum(np.sqrt(calculate_distance(data[np.where(matrix == i)],new_centres[i])))
    loss_K_plus_plus_function.append(loss)
print(loss_K_plus_plus_function)  


# In[12]:


plt.plot(x_temp,loss_K_plus_plus_function)
plt.show()

