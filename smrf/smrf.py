import numpy as np

from rasterio.transform import from_origin
from pandas import DataFrame

from scipy import sparse, interpolate
from scipy.sparse.linalg import lsqr

import struct

# from scipy.ndimage.morphology import grey_opening

from skimage.morphology import disk
from skimage.morphology.grey import opening

def smrf(x,y,z,cellsize=1,windows=5,slope_threshold=.15,elevation_threshold=.5,
         elevation_scaler=1.25,low_filter_slope=5,low_outlier_fill=False,return_AGH=False):
    """
    Simple Example:
    
    import smrf
    
    dtm, T, obj_grid, obj_vector = smrf.smrf(x,y,z,cellsize=1,windows=5,slope_threshold=.15)
    
    Parameters:
    - x,y,z are points in space (e.g., lidar points)
    - 'windows' is a scalar value specifying the maximum radius in pixels.  One can also 
                supply an array of specific radii to test.  Very often, increasing the 
                radius by one each time (as is the default) is unnecessary, especially 
                for EDA.  This is the most sensitive parameter.  Use a small value 
                (2-5 m) when removing small objects like trees.  Use a larger value (5-50)
                to remove larger objects like buildings.  Use the smallest value you
                can to avoid misclassifying true ground points as objects.  A
                small radius (5 pixels) and evaluating output is generally a good starting
                point.
    - 'slope_threshold' is a dz/dx value that controls the ground/object classification.
                A value of .15 to .2 is generally a good starting point.  Use a higher 
                value in steeper terrain to avoid misclassifying true ground points as
                objects.  Note, .15 equals a 15 percent (not degree!) slope.
    - 'elevation_threshold' is a value that controls final classification of object/ground
                and is specified in map units (e.g., meters or feet).  Any value within 
                this distance of the provisional DTM is considered a ground point.  A 
                value of .5 meters is generally a good starting point.
    - 'elevation_scaler' - allows elevation_threshold to increase on steeper slopes.
                The product of this and a slope surface are added to the elevation_threshold
                for each grid cell.  Set this value to zero to not use this paramater.
                A value of 1.25 is generally a good starting point.
    - 'low_filter_slope' controls the identification of low outliers.  Since SMRF builds
                its provisional DTM from the lowest point in a grid cell, low outliers can
                greatly affect algorithm performance.  The default value of 5 (corresponding
                to 500%) is a good starting point, but if your data has significant low outliers, 
                use a significantly higher value (50, 500) to remove these.
    - 'low_outlier_fill' removes and re-interpolates low outlier grid cells.  The default value
                is false, as most of the time the standard removal process works fine.
                
                
    Returns: dtm, transform, object_grid, object_vector
        
    - 'dtm' is a provisional ground surface created after processing.
    - 'T' is a rasterio Affine transformation vector for writing out the DTM using rasterio
    - 'obj_grid' is a boolean grid of the same size as DTM where 0s mark ground and 1s mark objects.
    - 'obj_vector' is a boolean vector/1D-array of the same size as x,y, and z, where 0s mark 
                ground and 1s mark objects.

                
    """         
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
    
    if return_AGH==False:  
        # Return the provisional surface, affine matrix, raster object cells
        # and boolean vector identifying object points from point cloud
        return Zpro,t,object_cells,is_object_point
    else:
        return Zpro,t,object_cells,is_object_point,z-elevation_values
    

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
        
# Given an image and accompanying affine transform, return the x_edges and
# y_edges for the image.  This is useful when passing specific edges to 
# the create_dem function.

def edges_from_IT(Image,Transform):
    r,c = np.shape(Image)[0],np.shape(Image)[1]
    x_edges = np.arange(c+1)
    y_edges = np.arange(r+1)
    x_edges, _ = Transform * np.array(list(zip(x_edges,np.zeros_like(x_edges)))).T
    _, y_edges = Transform * np.array(list(zip(np.zeros_like(y_edges),y_edges))).T
    
    return x_edges, y_edges


#%%
# Using scipy's binned statistic would be preferable here, but it doesn't do
# min/max natively, and is too slow when not cython.
# It would look like: 
# Z,xi,yi,binnum = stats.binned_statistic_2d(x,y,z,statistic='min',bins=(x_edge,y_edge))
def create_dem(x,y,z,cellsize=1,bin_type='max',inpaint=False,edges=None,use_binned_statistic=False):
    
    #x = df.x.values
    #y = df.y.values
    #z = df.z.values
    #resolution = 1
    #bin_type = 'max' 
    floor2 = lambda x,v: v*np.floor(x/v)
    ceil2 = lambda x,v: v*np.ceil(x/v)
    
    if edges is None:
        xedges = np.arange(floor2(np.min(x),cellsize)-.5*cellsize,
                           ceil2(np.max(x),cellsize) + 1.5*cellsize,cellsize)
        yedges = np.arange(ceil2(np.max(y),cellsize)+.5*cellsize,
                           floor2(np.min(y),cellsize) - 1.5*cellsize,-cellsize)
    else:
        xedges = edges[0]
        yedges = edges[1]
        out_of_range = (x < xedges[0]) | (x > xedges[-1]) | (y > yedges[0]) | (y < yedges[-1])
        x = x[~out_of_range]
        y = y[~out_of_range]
        z = z[~out_of_range]
        cellsize = np.abs(xedges[1]-xedges[0])
        
    nx, ny = len(xedges)-1,len(yedges)-1
    
    I = np.empty(nx*ny)
    I[:] = np.nan
    
    # Define an affine matrix, and convert realspace coordinates to integer pixel
    # coordinates
    t = rasterio.transform.from_origin(xedges[0], yedges[0], cellsize, cellsize)
    c,r = ~t * (x,y)
    c,r = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
    
    # Old way:
    # Use pixel coordiantes to create a flat index; use that index to aggegrate, 
    # using pandas.
    if use_binned_statistic:
        I = stats.binned_statistic_2d(x,y,z,statistic='min',bins=(xedges,yedges))
    else:        
        mx = pd.DataFrame({'i':np.ravel_multi_index((r,c),(ny,nx)),'z':z}).groupby('i')
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


def read_las(filename):
    """
    An LAS lidar file reader that outputs the point cloud into a Pandas 
    DataFrame.  It is written in pure Python, and relies only on common 
    scientific packages such as Numpy and Pandas.
    
    It does not yet work for LAZ or zLAS files.
    
    Simple Example:
    header, df = smrf.read_las('file.las')
                
    """    
    with open(filename,mode='rb') as file:
        data = file.read()
    
    # This dictionary holds the byte length of the point data (see minimum
    # PDRF Size given in LAS spec.)
    point_data_format_key = {0:20,1:28,2:26,3:34,4:57,5:63,6:30,7:36,8:38,9:59,10:67}
    
    # Read header into a dictionary
    header = {}
    header['file_signature'] = struct.unpack('<4s',data[0:4])[0].decode('utf-8')
    header['file_source_id'] = struct.unpack('<H',data[4:6])[0]
    header['global_encoding'] = struct.unpack('<H',data[6:8])[0]
    project_id = []
    project_id.append(struct.unpack('<L',data[8:12])[0])
    project_id.append(struct.unpack('<H',data[12:14])[0])
    project_id.append(struct.unpack('<H',data[14:16])[0])
    #Fix
    #project_id.append(struct.unpack('8s',data[16:24])[0].decode('utf-8').rstrip('\x00'))
    header['project_id'] = project_id
    del project_id
    header['version_major'] = struct.unpack('<B',data[24:25])[0]
    header['version_minor'] = struct.unpack('<B',data[25:26])[0]
    header['version'] = header['version_major'] + header['version_minor']/10
    header['system_id'] = struct.unpack('32s',data[26:58])[0].decode('utf-8').rstrip('\x00')
    header['generating_software'] = struct.unpack('32s',data[58:90])[0].decode('utf-8').rstrip('\x00')
    header['file_creation_day'] = struct.unpack('H',data[90:92])[0]
    header['file_creation_year'] = struct.unpack('<H',data[92:94])[0]
    header['header_size'] = struct.unpack('H',data[94:96])[0]
    header['point_data_offset'] = struct.unpack('<L',data[96:100])[0]
    header['num_variable_records'] = struct.unpack('<L',data[100:104])[0]
    header['point_data_format_id'] = struct.unpack('<B',data[104:105])[0]
    #print(header['point_data_format_id'])
    laz_format = False
    if header['point_data_format_id'] >= 128 and header['point_data_format_id'] <= 133:
        laz_format = True
        # error here?
        header['point_data_format_id'] =  header['point_data_format_id'] - 128
    if laz_format:
        raise ValueError('LAZ not yet supported.')
    try:
        format_length = point_data_format_key[header['point_data_format_id']]
    except:
        raise ValueError('Point Data Record Format',header['point_data_format_id'],'not yet supported.')
    if header['point_data_format_id'] >= 6:
        print('Point Data Formats 6-10 have recently been added to this reader.  Please check results carefully and report any suspected errors.')
    header['point_data_record_length'] = struct.unpack('<H',data[105:107])[0]
    header['num_point_records'] = struct.unpack('<L',data[107:111])[0]
    header['num_points_by_return'] = struct.unpack('<5L',data[111:131])
    header['scale'] = struct.unpack('<3d',data[131:155])
    header['offset'] = struct.unpack('<3d',data[155:179])
    header['minmax'] = struct.unpack('<6d',data[179:227]) #xmax,xmin,ymax,ymin,zmax,zmin
    end_point_data = len(data)
    
    # For version 1.3, read in the location of the point data.  At this time
    # no wave information will be read
    header_length = 227
    if header['version']==1.3:
        header['begin_wave_form'] = struct.unpack('<q',data[227:235])[0]
        header_length = 235
        if header['begin_wave_form'] != 0:
            end_point_data = header['begin_wave_form']

    # Pare out only the point data
    data = data[header['point_data_offset']:end_point_data]

    if header['point_data_format_id']==0:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2')])
    elif header['point_data_format_id']==1:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8')])
    elif header['point_data_format_id']==2:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('red','u2'),
                       ('green','u2'),('blue','u2')])
    elif header['point_data_format_id']==3:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('red','u2'),('green','u2'),('blue','u2')])
    elif header['point_data_format_id']==4:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('wave_packet_descriptor_index','u1'),('byte_offset','u8'),
                       ('wave_packet_size','u4'),('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==5:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('class','u1'),('scan_angle','u1'),
                       ('user_data','u1'),('point_source_id','u2'),('gpstime','f8'),
                       ('red','u2'),('green','u2'),('blue','u2'),
                       ('wave_packet_descriptor_index','u1'),('byte_offset','u8'),
                       ('wave_packet_size','u4'),('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==6:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8')])
    elif header['point_data_format_id']==7:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2')])        
    elif header['point_data_format_id']==8:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2'),
                       ('near_infrared','u2')])    
    elif header['point_data_format_id']==9:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('wave_packet_descriptor_index','u1'),
                       ('byte_offset','u8'),('wave_packet_size','u4'),
                       ('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])
    elif header['point_data_format_id']==10:
        dt = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('intensity', 'u2'),
                       ('return_byte','u1'),('mixed_byte','u1'),('class','u1'),
                       ('user_data','u1'),('scan_angle','u2'),('point_source_id','u2'),
                       ('gpstime','f8'),('red','u2'),('green','u2'),('blue','u2'),
                       ('near_infrared','u2'),('wave_packet_descriptor_index','u1'),
                       ('byte_offset','u8'),('wave_packet_size','u4'),
                       ('return_point_waveform_location','f4'),
                       ('xt','f4'),('yt','f4'),('zt','f4')])         
        
        
    # Transform to Pandas dataframe, via a numpy array
    data = pd.DataFrame(np.frombuffer(data,dt))
    data['x'] = data['x']*header['scale'][0] + header['offset'][0]
    data['y'] = data['y']*header['scale'][1] + header['offset'][1]
    data['z'] = data['z']*header['scale'][2] + header['offset'][2]

    def get_bit(byteval,idx):
        return ((byteval&(1<<idx))!=0);

    # Recast the mixed bytes as specified in the LAS specification
    if header['point_data_format_id'] < 6:
        data['return_number'] = 4 * get_bit(data['return_byte'],2).astype(np.uint8) + 2 * get_bit(data['return_byte'],1).astype(np.uint8) + get_bit(data['return_byte'],0).astype(np.uint8)
        data['return_max'] = 4 * get_bit(data['return_byte'],5).astype(np.uint8) + 2 * get_bit(data['return_byte'],4).astype(np.uint8) + get_bit(data['return_byte'],3).astype(np.uint8)
        data['scan_direction'] = get_bit(data['return_byte'],6)
        data['edge_of_flight_line'] = get_bit(data['return_byte'],7)
        del data['return_byte']
    else:
        data['return_number'] = 8 * get_bit(data['return_byte'],3).astype(np.uint8) + 4 * get_bit(data['return_byte'],2).astype(np.uint8) + 2 * get_bit(data['return_byte'],1).astype(np.uint8) + get_bit(data['return_byte'],0).astype(np.uint8)
        data['return_max'] = 8 * get_bit(data['return_byte'],7).astype(np.uint8) + 4 * get_bit(data['return_byte'],6).astype(np.uint8) + 2 * get_bit(data['return_byte'],5).astype(np.uint8) + get_bit(data['return_byte'],4).astype(np.uint8)
        # data['scan_direction'] = get_bit(data['return_byte'],6)
        # data['edge_of_flight_line'] = get_bit(data['return_byte'],7)
        del data['return_byte']        
    if header['point_data_format_id'] >= 6:
        data['classification_bit_synthetic'] = get_bit(data['mixed_byte'],0)
        data['classification_bit_keypoint'] = get_bit(data['mixed_byte'],1)
        data['classification_bit_withheld'] = get_bit(data['mixed_byte'],2)
        data['classification_bit_overlap'] = get_bit(data['mixed_byte'],3)
        data['scanner_channel'] = 2 * get_bit(data['mixed_byte'],5).astype(np.uint8) + 1 * get_bit(data['mixed_byte'],4).astype(np.uint8)
        data['scan_direction'] = get_bit(data['mixed_byte'],6)
        data['edge_of_flight_line'] = get_bit(data['mixed_byte'],7)
        del data['mixed_byte']
    

    
    return header,data
