#!/usr/bin/python
# -*- coding: utf-8 -*-
import gdal
import numpy
from numpy  import *
import math
from numbapro import cuda
import CUDA_hillshade
import time

start = time.clock()
dsm_file = r"F:\Lidar\outputShanghai/I001_001_proj.tif"
stringIndex = int(dsm_file.rfind("/"))
workSpacePath = dsm_file[0:int(stringIndex)]
inputRaster = dsm_file[stringIndex + 1:]
fn=workSpacePath+ '\\'+ inputRaster
ds = gdal.Open(fn)
a=gdal.Dataset.ReadAsArray(ds)
b=a.reshape(-1)
geoTransform=ds.GetGeoTransform()
geoTransform=numpy.array(geoTransform)
geoT=geoTransform.reshape(-1)
# geoTransform2_dev = cuda.to_device(geoT) #地理坐标数据读入GPU
if ds is None:
    print 'open failed'
cols = ds.RasterXSize
rows = ds.RasterYSize
x_cellsize=math.fabs(geoTransform[5])
y_cellsize=geoTransform[1]
slope=numpy.zeros_like(a,dtype=numpy.float64)
Hillshade=numpy.zeros_like(a,dtype=numpy.int16)#dtype=numpy.int16
Hillshade=Hillshade.reshape(-1)
# Shadow=numpy.zeros_like(a,dtype=numpy.int16)
# Shadow=Shadow.reshape(-1)
# shadow=numpy.zeros_like(Shadow)
# Shadow2_dev = cuda.to_device(Shadow,stream)
xrate=numpy.zeros_like(a,dtype=numpy.float64)
yrate=numpy.zeros_like(a,dtype=numpy.float64)
latMax = geoTransform[3]
latMin = latMax+rows*geoTransform[5]
lonMin = geoTransform[0]
lonMax = lonMin+cols*geoTransform[1]
radius=math.sqrt(math.pow(geoTransform[5],2)+math.pow(geoTransform[1],2))
#radius=numpy.array(radius)
#radius_dev = cuda.to_device(radius)
MaxElevationValue=a.max()
altitude = 45
Zen_deg = 90 - altitude
zen_rad = Zen_deg * math.pi / 180
Azimuth = 315
Theta = math.radians(Azimuth)
Phi = math.radians(altitude)
Azimuth_math = 360 - Azimuth + 90
if Azimuth_math > 360:
    Azimuth_math = Azimuth_math - 360
azimuth_rad = Azimuth_math * math.pi /180
Zfactor=0.00001036
rAD_90 = math.radians(90.0)
rAD_180 = math.radians(180.0)
rAD_270 = math.radians(270.0)
# parameter=[cols,rows,radiusInterval,Azimuth,Altitude,RAD_90,RAD_180,RAD_270,LatMax,LatMin,LonMax,LonMin,maxElevationValue,Zen_rad,Azimuth_rad]
# parameter=numpy.array(parameter)
# parameter=parameter.reshape(-1)
# parameter2_dev = cuda.to_device(parameter)

for i in xrange(1, rows - 2):
    for j in xrange(1, cols - 2):
        xrate[i,j]= ((a[i-1, j+1] + a[i+1, j+1] + 2 * a[i,j +1]) - (a[i-1, j-1] + a[i+1, j-1] + 2 * a[i, j-1])) / (8*x_cellsize)
        yrate[i,j]= ((a[i+1, j-1] + a[i+1, j+1] + 2 * a[i+1,j]) - (a[i-1,j-1] + a[i-1, j+1] + 2 * a[i-1,j])) / (8*y_cellsize)
        rise_run = math.sqrt(((math.pow(xrate[i,j],2)+ math.pow(yrate[i,j],2))))
        slope[i, j] = math.atan(Zfactor * rise_run)
xrate=xrate.reshape(-1)
yrate=yrate.reshape(-1)
slope=slope.reshape(-1)
end1=time.clock()

#定义GPU核函数
#blockdim = (32, 8)维度定义
#griddim = (32,16)
griddim=40
blockdim=1024
stream = cuda.stream()
with stream.auto_synchronize():
    aNumpy2_dev = cuda.to_device(b,stream=stream)  # 将DSM数据读入GPU
    xrate2_dev = cuda.to_device(xrate,stream=stream)
    yrate2_dev = cuda.to_device(yrate,stream=stream)
    slope2_dev = cuda.to_device(slope,stream=stream)  # slope数据读入gpu
    Hillshade2_dev =cuda.to_device(Hillshade,stream=stream)
    CUDA_hillshade.hillshadeshadow[griddim,blockdim,stream](geoT[0],geoT[1],geoT[2],geoT[3],geoT[4],geoT[5],aNumpy2_dev,slope2_dev,cols,rows,radius,Theta,Phi,rAD_90,rAD_180,rAD_270,latMax,latMin,lonMax,lonMin,MaxElevationValue,zen_rad,azimuth_rad,xrate2_dev,yrate2_dev,Hillshade2_dev)
    hillshade=Hillshade2_dev.copy_to_host(stream=stream)
    #Hillshade2_dev.to_host(stream=stream)
hillshade=hillshade.reshape(rows,cols)
hillshade=numpy.array(hillshade)
end2=time.clock()
print("CPU运行时间%.03f seconds"%(end1-start))
print("GPU运行时间%.03f seconds"%(end2-end1))
driver = ds.GetDriver()
filename=r'F:\Lidar\outputShanghai\hillshade1.tif'
outDataset=driver.Create(filename, ds.RasterXSize, ds.RasterYSize,1,gdal.GDT_Float32)
outBand = outDataset.GetRasterBand(1)
outBand.WriteArray(hillshade, 0, 0)
outDataset.SetGeoTransform(geoTransform)
proj = ds.GetProjection()
outDataset.SetProjection(proj)