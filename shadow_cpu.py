# -*- coding: utf-8 -*-
import gdal
import numpy
from numpy  import *
import math
import time
start = time.clock()
dsm_file = r"F:\output/test.tif"#stringIndex = int(dsm_file.rfind("/"))
workSpacePath = dsm_file[0:int(stringIndex)]
inputRaster = dsm_file[stringIndex + 1:]
fn=workSpacePath+ '\\'+ inputRaster
ds = gdal.Open(fn)
a=gdal.Dataset.ReadAsArray(ds)
geoTransform=ds.GetGeoTransform()
if ds is None:
    print 'open failed'
cols = ds.RasterXSize
rows = ds.RasterYSize
x_cellsize=math.fabs(geoTransform[5])
y_cellsize=geoTransform[1]
slope=numpy.zeros_like(a)
hillshadeaspect=numpy.zeros_like(a)
hillshade=numpy.zeros_like(a)
shadow=numpy.zeros_like(a)
shadow[:,:]=255
latitudeResolution =geoTransform[5]#纬度分辨率，注意纬度分辨率是负值
longitudeResolution = geoTransform[1]#经度分辨率
LatMax = geoTransform[3]#最大纬度
LatMin = LatMax+rows*latitudeResolution#最小纬度
LonMin = geoTransform[0]#最小经度
LonMax = LonMin+cols*longitudeResolution#最大经度
radiusInterval=math.sqrt(math.pow(latitudeResolution,2)+math.pow(longitudeResolution,2))#逐增半径
maxElevationValue=a.max()#研究范围地表最大高程值
Altitude = 45#太阳高度角α
Zen_deg = 90 - Altitude
Zen_rad = Zen_deg * math.pi / 180#高度角转弧度
Azimuth = 50#太阳方位角θ
Azimuth_math = 360 - Azimuth + 90
if Azimuth_math > 360:
    Azimuth_math = Azimuth_math - 360
Azimuth_rad = Azimuth_math * math.pi / 180#方位角转弧度
Zfactor=0.00001036#参考Arcgis
RAD_90 = math.radians(90.0)
RAD_180 = math.radians(180.0)
RAD_270 = math.radians(270.0)
def SpheresgetPoint3D(Azi,Alt,radiu):#迭代半径与太阳高度角方位角计算光线对应球面坐标xyz
    theta=math.radians(Azi)
    phi = math.radians(Alt)
    _y=math.sqrt(math.pow(radiu, 2) - math.pow(radiu* math.cos(phi), 2))
    r0=math.sqrt(math.pow(radiu, 2) - math.pow(_y, 2))
    _b = r0 * math.cos(theta)
    _z = math.sqrt(math.pow(r0, 2) - math.pow(_b, 2))
    _x = math.sqrt(math.pow(r0, 2) - math.pow(_z, 2))
    if theta<= RAD_90:
        _z *= -1.0
    elif theta <= RAD_180:
        _x *= -1.0
        _z *= -1.0
    elif theta <= RAD_270:
        _x *= -1.0
    if (phi >= 0) :
        _y = math.fabs(_y)
    else :
        _y = math.fabs(_y) * -1
    points=[]
    points.append(_x)
    points.append(_y)
    points.append(_z)
    return points

for i in xrange(1, rows - 1):
    for j in xrange(1, cols - 1):
        centerlongitude = geoTransform[0] + j * geoTransform[1] + i * geoTransform[2]#待判断点经度
        centerlatitude = geoTransform[3] + j * geoTransform[4] + i * geoTransform[5]#待判断点纬度
        centerelevation = a[i, j]#待判断点高度
        # 初始化迭代半径
        r = radiusInterval
        while True:
            points = SpheresgetPoint3D(Azimuth, Altitude, r)
            pointlatitude = centerlatitude + points[0]#光线点纬度
            pointlongitude = centerlongitude - points[2]#光线点经度
            # 进行遮挡检测
            if pointlatitude > LatMax or pointlatitude < LatMin or pointlongitude > LonMax or pointlongitude < LonMin:
                shadow[i, j] = 255
                break
            try:
                pointelevation = a[(LatMax - pointlatitude) / (-latitudeResolution), (pointlongitude - LonMin) / longitudeResolution]
            except Exception, e:
                print e.message
                r += radiusInterval
                continue
            rayelevation = centerelevation + points[1] / 0.00001036
			if rayelevation > maxElevationValue:
                shadow[i, j] = 255
                break
            if rayelevation < pointelevation:
                shadow[i, j] = 0
                break
            r += radiusInterval
		#下面是计算hillshade部分，没有需要的可以不看下面这部分
        if shadow[i,j]==0:
            hillshade[i,j]=shadow[i,j]
        else:
            xrate=((a[i-1,j+1]+a[i+1,j+1]+2*a[i,j+1])-(a[i-1,j-1]+a[i+1,j-1]+2*a[i,j-1]))/(8*x_cellsize)
            yrate=((a[i+1,j-1]+a[i+1,j+1]+2*a[i+1,j])-(a[i-1,j-1]+a[i-1,j+1]+2*a[i-1,j]))/(8*y_cellsize)
            rise_run=sqrt(((xrate*xrate)+(yrate*yrate)))
            slope[i,j]=math.atan(Zfactor*rise_run)
            if xrate!=0:
                hillshadeaspect[i,j]=math.atan2(yrate, -xrate)
                if hillshadeaspect[i,j]<0:
                    hillshadeaspect[i, j] = 2*math.pi + hillshadeaspect[i, j]
            if xrate==0:
                if yrate>0:
                    hillshadeaspect[i,j]=math.pi/2
                elif yrate<0:
                    hillshadeaspect[i,j] = 2*math.pi- math.pi/2
                elif yrate == 0:
                    hillshadeaspect[i, j]=hillshadeaspect[i,j]
            hillshade[i,j] =255.0 * ((cos(Zen_rad) * cos(slope[i,j])) +(sin(Zen_rad) * sin(slope[i,j]) * cos(Azimuth_rad - hillshadeaspect[i,j])))
            if hillshade[i,j]<0:
                hillshade[i, j]=0
            hillshade[i,j]=int(hillshade[i,j])
        print"!"
    print"!!"
end=time.clock()
print ("运行时间%.03f seconds" %(end-start))
#计算结果导出为栅格文件
driver = ds.GetDriver()
filename=r'F:\Lidar\output\hillshadeshadow.tif'
outDataset=driver.Create(filename, ds.RasterXSize, ds.RasterYSize, 1,gdal.GDT_Float32)
outBand = outDataset.GetRasterBand(1)
outBand.WriteArray(hillshade, 0, 0)
outDataset.SetGeoTransform(geoTransform)
proj = ds.GetProjection()
outDataset.SetProjection(proj)
