import numpy as np

from rasterio.transform import from_origin
from pandas import DataFrame

from scipy import sparse, interpolate
from scipy.sparse.linalg import lsqr

# from scipy.ndimage.morphology import grey_opening

from skimage.morphology import disk
from skimage.morphology.grey import opening

'''
x,y,z are points in space (e.g., lidar points)
windows is a scalar value specifying the maximum radius in pixels.  One can also 
supply an array of specific radii to test.  Very often, increasing the radius by 
one each time (as is the default) is unnecessary, especially for EDA.
Final classification of points is done using elevation_threshold and elevation_scaler.
points are compared against the provisional surface with a threshold modulated by the 
scaler value.  However, often the provisional surface (itself interpolated) works
quite well.  
Two additional parameters are being test to assist in low outlier removal.
low_filter_slope provides a slope value for an inverted surface.  Its default
value is 5 (meaning 500% slope).  However, we have noticed that in very rugged 
and forested terrain, that an even larger value may be necessary.  Alternatively,
low noise can be scrubbed using other means, and then this value can be set to a very high 
value to avoid its use entirely.  A second parameter (low_outlier_fill) will 
remove the points from the provisional DTM, and then fill them in before the main
body of the SMRF algorithm proceeds.  This should aid in preventing the "damage"
to the DTM that can happen when low outliers are present.
Returns Zpro,t,object_cells,is_object_point.
'''

def smrf(x,y,z,cellsize=1,windows=18,slope_threshold=.15,elevation_threshold=.5,
         elevation_scaler=1.25,low_filter_slope=5,low_outlier_fill=False):

    if np.isscalar(windows):
        windows = np.arange(windows) + 1
    
    Zmin,t = create_dem(x,y,z,cellsize=cellsize,bin_type='min');
    is_empty_cell = np.isnan(Zmin)
    Zmin = inpaint_nans_by_springs(Zmin)
    low_outliers = progressive_filter(-Zmin,np.array([1]),cellsize,slope_threshold=low_filter_slope); 
    
    # perhaps best to remove and interpolate those low values before proceeding?
    if low_outlier_fill:
        Zmin[low_outliers] = np.nan
        Zmin = inpaint_nans_by_springs(Zmin)
    
    # This is the main crux of the algorithm
    object_cells = progressive_filter(Zmin,windows,cellsize,slope_threshold);
    
    # Create a provisional surface
    Zpro = Zmin
    del Zmin
    # For the purposes of returning values to the user, an "object_cell" is
    # any of these: empty cell, low outlier, object cell
    object_cells = is_empty_cell | low_outliers | object_cells
    Zpro[object_cells] = np.nan
    Zpro = inpaint_nans_by_springs(Zpro)
    
    # Use provisional surface to interpolate a height at each x,y point in the
    # point cloud.  The original SMRF used a splined cubic interpolator. 
    col_centers = np.arange(0.5,Zpro.shape[1]+.5)
    row_centers = np.arange(0.5,Zpro.shape[0]+.5)

    # RectBivariateSpline Interpolator
    c,r = ~t * (x,y)
    f1 = interpolate.RectBivariateSpline(row_centers,col_centers,Zpro)
    elevation_values = f1.ev(r,c)
    
    # Calculate a slope value for each point.  This is used to apply a some "slop"
    # to the ground/object ID, since there is more uncertainty on slopes than on
    # flat areas.
    gy,gx = np.gradient(Zpro,cellsize)
    S = np.sqrt(gy**2 + gx**2)
    del gy,gx
    f2 = interpolate.RectBivariateSpline(row_centers,col_centers,S)
    del S
    slope_values = f2.ev(r,c)
    
    # Use elevation and slope values and thresholds interpolated from the 
    # provisional surface to classify as object/ground
    required_value = elevation_threshold + (elevation_scaler * slope_values)
    is_object_point = np.abs(elevation_values - z) > required_value
    
    # Return the provisional surface, affine matrix, raster object cells
    # and boolean vector identifying object points from point cloud
    return Zpro,t,object_cells,is_object_point
    

def progressive_filter(Z,windows,cellsize=1,slope_threshold=.15):
    last_surface = Z.copy()
    elevation_thresholds = slope_threshold * (windows * cellsize)  
    is_object_cell = np.zeros(np.shape(Z),dtype=np.bool)
    for i,window in enumerate(windows):
        elevation_threshold = elevation_thresholds[i]
        this_disk = disk(window)
        if window==1:
            this_disk = np.ones((3,3),dtype=np.uint8)
        this_surface = opening(last_surface,disk(window)) 
        is_object_cell = (is_object_cell) | (last_surface - this_surface > elevation_threshold)
        if i < len(windows) and len(windows)>1:
            last_surface = this_surface.copy()
    return is_object_cell

    
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
            

def inpaint_nans_by_springs(A):
    
    neighbors = 4
    inplace = False

    m,n = np.shape(A)
    nanmat = np.isnan(A)

    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)
    
    r,c = np.unravel_index(nan_list,(m,n))
    
    num_neighbors = neighbors
    neighbors = np.array([[0,1],[0,-1],[-1,0],[1,0]]) #r,l,u,d

    neighbors = np.vstack([np.vstack((r+i[0], c+i[1])).T for i in neighbors])
    del r,c
    
    springs = np.tile(nan_list,num_neighbors)
    good_rows = (np.all(neighbors>=0,1)) & (neighbors[:,0]<m) & (neighbors[:,1]<n)
    
    neighbors = np.ravel_multi_index((neighbors[good_rows,0],neighbors[good_rows,1]),(m,n))
    springs = springs[good_rows]
    
    springs = np.vstack((springs,neighbors)).T
    del neighbors,good_rows
    
    springs = np.sort(springs,axis=1)
    springs = unique_rows(springs)
    
    n_springs = np.shape(springs)[0]
    
    i = np.tile(np.arange(n_springs),2)
    springs = springs.T.ravel()
    data = np.hstack((np.ones(n_springs,dtype=np.int8),-1*np.ones(n_springs,dtype=np.int8)))
    springs = sparse.coo_matrix((data,(i,springs)),(n_springs,m*n),dtype=np.int8).tocsr()
    del i,data
    
    rhs = -springs[:,known_list] * A[np.unravel_index(known_list,(m,n))]
    results = lsqr(springs[:,nan_list],rhs)[0]       

    if inplace:
        A[np.unravel_index(nan_list,(m,n))] = results
    else:
        B = A.copy()
        B[np.unravel_index(nan_list,(m,n))] = results
        return B
    
# Finite difference approximation
def inpaint_nans_by_fda(A,fast=True):

    inplace=False

    m,n = np.shape(A)
    nanmat = np.isnan(A)

    nan_list = np.flatnonzero(nanmat)
    known_list = np.flatnonzero(~nanmat)
    
    index = np.arange(m*n,dtype=np.int64).reshape((m,n))
    
    i = np.hstack( (np.tile(index[1:-1,:].ravel(),3),
                    np.tile(index[:,1:-1].ravel(),3)
                    ))
    j = np.hstack( (index[0:-2,:].ravel(),
                    index[2:,:].ravel(),
                    index[1:-1,:].ravel(),
                    index[:,0:-2].ravel(),
                    index[:,2:].ravel(),
                    index[:,1:-1].ravel()
                    ))
    data = np.hstack( (np.ones(2*n*(m-2),dtype=np.int64),
                       -2*np.ones(n*(m-2),dtype=np.int64),
                       np.ones(2*m*(n-2),dtype=np.int64),
                       -2*np.ones(m*(n-2),dtype=np.int64)
                       ))
    if fast==True:
        goodrows = np.in1d(i,index[ndi.binary_dilation(nanmat)])
        i = i[goodrows]
        j = j[goodrows]
        data = data[goodrows]
        del goodrows

    fda = sparse.coo_matrix((data,(i,j)),(m*n,m*n),dtype=np.int8).tocsr()
    del i,j,data,index
    
    rhs = -fda[:,known_list] * A[np.unravel_index(known_list,(m,n))]
    k = fda[:,np.unique(nan_list)]
    k = k.nonzero()[0]
    a = fda[k][:,nan_list]
    results = sparse.linalg.lsqr(a,rhs[k])[0]

    if inplace:
        A[np.unravel_index(nan_list,(m,n))] = results
    else:
        B = A.copy()
        B[np.unravel_index(nan_list,(m,n))] = results
    return B
        
def create_dem(x,y,z,cellsize=1,bin_type='max',use_binned_statistic=False,inpaint=False):
    
    floor2 = lambda x,v: v*np.floor(x/v)
    ceil2 = lambda x,v: v*np.ceil(x/v)
    
    
    xedges = np.arange(floor2(np.min(x),cellsize)-.5*cellsize,
                       ceil2(np.max(x),cellsize) + 1.5*cellsize,cellsize)
    yedges = np.arange(ceil2(np.max(y),cellsize)+.5*cellsize,
                       floor2(np.min(y),cellsize) - 1.5*cellsize,-cellsize)
    nx, ny = len(xedges)-1,len(yedges)-1
    
    I = np.empty(nx*ny)
    I[:] = np.nan
    
    # Define an affine matrix, and convert realspace coordinates to integer pixel
    # coordinates
    t = from_origin(xedges[0], yedges[0], cellsize, cellsize)
    c,r = ~t * (x,y)
    c,r = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
    
    # Old way:
    # Use pixel coordiantes to create a flat index; use that index to aggegrate, 
    # using pandas.
    if use_binned_statistic:
        I = stats.binned_statistic_2d(x,y,z,statistic='min',bins=(xedges,yedges))
    else:        
        mx = DataFrame({'i':np.ravel_multi_index((r,c),(ny,nx)),'z':z}).groupby('i')
        del c,r
        if bin_type=='max':
            mx = mx.max()
        elif bin_type=='min':
            mx = mx.min()
        else:
            raise ValueError('This type not supported.')
        
        I.flat[mx.index.values] = mx.values
        I = I.reshape((ny,nx))
        
    if inpaint==True:
        I = inpaint_nans_by_springs(I)
    
    return I,t