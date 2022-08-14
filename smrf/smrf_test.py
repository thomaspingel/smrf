# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:40:42 2021

@author: Thomas Pingel
"""

#%%

import numpy as np


#%% Development Test

x,y,z = 10 * np.random.rand(3,100)
Zpro,t,object_cells,is_object_point,extras = smrf(x,y,z,return_extras=True)


#%% Package Test

import smrf

x,y,z = 10 * np.random.rand(3,100)
Zpro,t,object_cells,is_object_point = smrf.smrf(x,y,z)


#%%


