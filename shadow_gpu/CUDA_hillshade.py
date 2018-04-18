# -*- coding: utf-8 -*-
from numbapro import cuda
import math

@cuda.autojit#(void(float64,float64,float64,float64,float64,float64,float32[:],float32[:],int16,int16,float64,float32,float32,float32,float32,float32,float64,float64,float64,float64,float32,float32,float32,float64[:],float64[:],int16[:],))
def hillshadeshadow(geo0,geo1,geo2,geo3,geo4,geo5,a,slope,col,row,radiusInterval,theta,phi,RAD_90,RAD_180,RAD_270,LatMax,LatMin,LonMax,LonMin,maxElevationValue,Zen_rad,Azimuth_rad,xrate,yrate,hillshade):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    x = tx + bx * bw
    for ii in range(x, col*row-1,40*1024):
        centerlongitude=geo0+ (ii%col)* geo1 + (ii/col) * geo2
        centerlatitude=geo3 + (ii%col) * geo4 + (ii/col) * geo5
        #r逐增半径
        r =radiusInterval
        centerelevation=a[ii]#待计算点的高程
        while True:
            #该太阳光线方向上某一点
            y=math.sqrt(math.pow(r, 2) - math.pow(r * math.cos(phi), 2))
            r0 =math.sqrt(math.pow(r, 2) - math.pow(y, 2))
            b =(math.sqrt(math.pow(r, 2) - math.pow(y, 2)))* math.cos(theta)
            z = math.sqrt(math.pow(r0, 2) - math.pow(b, 2))
            s = math.sqrt(math.pow(r0, 2) - math.pow(z, 2))
            if theta <= RAD_90:
                z *= -1.0
            elif theta <= RAD_180:
                s *= -1.0
                z *= -1.0
            elif theta <= RAD_270:
                s *= -1.0
            if (phi >= 0):
                y = math.fabs(y)
            else:
                y = math.fabs(y) * -1
            pointlatitude = centerlatitude+s
            pointlongitude = centerlongitude-z
            # 判断该点是否超出边界
            if pointlatitude >LatMax or pointlatitude < LatMin or  pointlongitude >LonMax or  pointlongitude <LonMin:
                shadow = 255
                break
            # 获取ray elevation
            # metersResolution=math.sqrt(math.pow((pointlatitude-centerlatitude)*latitudeResolutionmeter,2)+math.pow((pointlongitude-centerlongitude)*longitudeResolutionmeter,2))
            rayelevation = centerelevation+(y/0.00001036)  # ((points[1]/r) * metersResolution)
            # 异常检测该点是否有值
            m=math.fabs((LatMax-pointlatitude)/(geo5))
            mm=int(m)
            n=(pointlongitude-LonMin)/(geo1)
            nn=int(n)
            k =int(((mm*col) + nn))
            pointelevation=a[k]
            r+=radiusInterval
            if rayelevation<pointelevation:
                shadow= 0
                break
            if rayelevation > maxElevationValue:
                shadow=255
                break

        #计算hillshadow
        if shadow == 0:
            hillshade[ii] = shadow
        else:
            if xrate[ii] != 0:
                hillshadeaspect = math.atan2(yrate[ii], -xrate[ii])
                if hillshadeaspect < 0:
                    hillshadeaspect = 2 * math.pi + hillshadeaspect
            else:
                if yrate[ii] > 0:
                    hillshadeaspect= math.pi/2
                elif yrate[ii]  < 0:
                    hillshadeaspect = 2 * math.pi - math.pi/2
                elif yrate[ii]  == 0:
                    hillshadeaspect= hillshadeaspect
            hillshade[ii] = 255.0 * ((math.cos(Zen_rad) * math.cos(slope[ii])) + (math.sin(Azimuth_rad) * math.sin(slope[ii]) * math.cos(Azimuth_rad - hillshadeaspect)))
            if hillshade[ii] < 0:
                hillshade[ii] = 0
            hillshade[ii]=(hillshade[ii])
    cuda.syncthreads()

