# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:40:42 2021

@author: Thomas Pingel
"""

#%%

import numpy as np
import pdal
import pandas as pd

#%% Development Test.  Does this even run?

x,y,z = 10 * np.random.rand(3,100)
Zpro,t,object_cells,is_object_point,extras = smrf(x,y,z,return_extras=True)


#%% Better test on real data

fn_in = '../../smrf_data/drillfield.las'
pipeline = pdal.Reader(fn_in).pipeline()
pipeline.execute()
arr = pipeline.arrays[0]

Zpro,t,object_cells,is_object_point,extras = smrf(arr['X'],arr['Y'],arr['Z'],
                                                  cellsize=1,windows=np.array([1,2,3,5,10,15,20,30,70]),return_extras=True)

df = pd.DataFrame({'x':arr['X'],'y':arr['Y'],'z':arr['Z'],'ground':2*(1-is_object_point),'dropped':extras['when_dropped']})

df.to_csv('test.csv',index=False)


#%%


#%% Package Test

import smrf

x,y,z = 10 * np.random.rand(3,100)
Zpro,t,object_cells,is_object_point,extras = smrf.smrf(x,y,z,return_extras=True)


#%%

fn_in = '../../smrf_data/drillfield.las'
pipeline = pdal.Reader(fn_in).pipeline()
pipeline.execute()
arr = pipeline.arrays[0]

Zpro,t,object_cells,is_object_point,extras = smrf.smrf(arr['X'],arr['Y'],arr['Z'],cellsize=1,windows=5,return_extras=True)


df = pd.DataFrame({'x':arr['X'],'y':arr['Y'],'z':arr['Z'],'ground':2*(1-is_object_point),'dropped':np.round(extras['when_dropped'],decimals=1)})

df.to_csv('test.csv',index=False)

