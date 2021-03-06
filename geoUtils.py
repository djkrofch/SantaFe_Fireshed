# ------- Notebook config
import matplotlib.colors
import matplotlib.pyplot as plt

# ------- Load dependencies
import pandas as pd
import numpy as np
import random
import fiona
from itertools import chain
import seaborn as sns
import gdal, os, osr, warnings
from numpy.lib.stride_tricks import as_strided

### Raster funcs
def importRaster(rasterPath, **kwargs):
    # Open and read in the raster as an array
    raster_ds = gdal.Open(rasterPath)
    rastermap = raster_ds.ReadAsArray()
    
    # Set the default data type to 'float'
    if 'dtype' not in kwargs:
        dtype = 'float'
    rastermap = rastermap.astype(dtype)
    
    # If specified, set the no data value to NaN
    if 'noData' in kwargs:
        rastermap[rastermap == noData] = np.nan
    return rastermap
        
def plotRaster(image, ax=None, *args, **kwargs):

    # Grab figure axes if none stated
    if ax == None:
         ax = plt.gca()
                   
    # Normalize color scheme
    if 'norm' not in kwargs:
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if vmin is None:
            vmin = np.min(image) # or what ever
        if vmax is None:
            vmax = np.max(image)
        norm = matplotlib.colors.Normalize(vmin, vmax)
        kwargs['norm'] = norm

    #ax.figure.canvas.draw() # if you want to force a re-draw
    ax.imshow(image, *args, **kwargs)
    # Setup axes
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
	
def saveAsGeoTiff(spatialRaster, ndarray, outputFileName, epsg):
	templatedf = gdal.Open(spatialRaster)
	template = templatedf.ReadAsArray()
	driver = gdal.GetDriverByName('GTiff')
	outputRaster = driver.Create(outputFileName,
								 template.shape[1],
								 template.shape[0],
								 1, gdal.GDT_Int32)

	srs = osr.SpatialReference()
	srs.ImportFromEPSG(epsg)
	dest_wkt = srs.ExportToWkt()
	outputRaster.SetGeoTransform(templatedf.GetGeoTransform())
	outputRaster.GetRasterBand(1).WriteArray(ndarray)
	outputRaster.SetProjection(dest_wkt)
	outputRaster.FlushCache()   
	
### Vector funcs
def getShpGeom(shapefile):
    shp = fiona.open(shapefile)
    bds = shp.bounds
    shp.close()
    padLON = ((bds[0] - bds[2]) / 2) * 0.05
    padLAT = ((bds[1] - bds[3]) / 2) * 0.05
    ll = (bds[0] + padLON, bds[1] + padLAT)
    ur = (bds[2] - padLON, bds[3] - padLAT)
    midlat = (bds[1] + bds[3]) / 2
    midlon = (bds[0] + bds[2]) / 2
    coords = list(chain(ll, ur))
    return coords, midlat, midlon