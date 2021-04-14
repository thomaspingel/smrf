# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:40:42 2021

@author: Thomas Pingel
"""

#%%

import numpy as np
import smrf
import matplotlib.pyplot as plt

x,y,z = 10 * np.random.rand(3,100)
Zpro,t,object_cells,is_object_point = smrf.smrf(x,y,z)