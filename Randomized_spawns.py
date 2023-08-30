# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:01:57 2022

@author: thijs
"""

import numpy as np
import random

def random_spawns(agents): 
    x = []
    y = []
    z = []
 
    #map dimensions
    c = 11
    d = 24
    
    #open file containing map
    with open('C:/Users/thijs/OneDrive/Bureaublad/Bio_Based/instance2.txt') as f:
        [x.append(line) for line in f.readlines()]

    f.close()
    #find allowable starting locations in first two columns and last two rows of map
    for i in np.arange(1, c-1, 1):
        for j in np.arange(1,3):
            y.append([i, j])
    for i in np.arange(1, c-1, 1):
        for j in np.arange(d - 3, d-1, 1):
            z.append([i, j])
    #open file to wirte random start and goal locs in
    with open('C:/Users/thijs/OneDrive/Bureaublad/Bio_Based/Assignment.txt', 'w') as f:
        for i in range(len(x)):
            f.write(x[i])
        f.write('\n')
        f.write(str(agents))
        for ii in range(agents):
            a = random.randint(1, len(y)-1)
            b = random.randint(1, len(z)-1)
            #randomly select a start and goal loc from the allowable locations
            start_loc = y[a-1]
            goal_loc = z[b-1]
            y.remove(y[a-1])
            z.remove(z[b-1])
            
            #write start and goal locs into assignment file
            txt = str(start_loc[0])
            txt1 = str(start_loc[1])
            txt2 = str(goal_loc[0])
            txt3 = str(goal_loc[1])
            f.write('\n')
            f.write(txt)
            f.write(' ')
            f.write(txt1)
            f.write(' ')
            f.write(txt2)
            f.write(' ')
            f.write(txt3)