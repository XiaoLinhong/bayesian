import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

mVtoPPM = 1/2.5 # signal to concentration, ppm/mV (Figarro sensor, IBM)
SCFHToPPM = 470/60 # ppm/SCFH

obsNoise = 0.5 # estimated from blank test, in ppm
obsNoiseRatio = 0.05 # estimated from sensor's datasheet
mdlErrorRatio = 0.2 # estimate (you can vary this, usually 0.2-0.4)

# 观测数据
interval = 30*60 # interval to perform plume reconstruction, in seconds
frequency = 0.5  # Sensor frequency (Hz): 两分钟一次

# 可能的排放强度
leakRates = np.arange(1,41) # Range of leak rate, in SCFH
## 目标源的海拔高度
hgt = xr.open_dataset('source_area.nc')['hgt']
ds = 0.3 # 目标网格宽度

## 观测点位信息
meta = pd.read_csv('sensor.csv', index_col=0)
meta.columns = meta.columns.str.replace(' ','') # 删除列名的空格

## 观测数据
obsData = pd.read_csv('2017-07-17_20-57-10.csv', skiprows=[1, 2, 3])
obsData.columns = obsData.columns.str.replace(' ','') # 删除列名的空格

## 模拟数据
mdlFile = xr.open_dataset('conc.nc')
mdlData = mdlFile['conc'].values
mdlX = mdlFile['x'].values
mdlY = mdlFile['y'].values
mdlZ = mdlFile['z'].values
dx = mdlX[1] - mdlX[0]
dy = mdlY[1] - mdlY[0]
dz = mdlZ[1] - mdlZ[0]
xBeg = mdlX[0] - dx/2
yBeg = mdlY[0] - dy/2
zBeg = mdlZ[0] - dz/2

# 处理背景浓度问题
sensorIDs = []
for sensorID in meta.index:
    if sensorID in obsData:
        sensorIDs.append(sensorID)
        obsData[sensorID] -= obsData[sensorID].quantile(0.05) # Background concentration (5th percentile)

        obsData[sensorID] = obsData[sensorID]*mVtoPPM # 单位转换
        obsData = obsData.replace(np.nan, 0) # 处理缺省值
        obsData[sensorID] = obsData[sensorID].where(obsData[sensorID]>0, 0)

nleak = len(leakRates)
nSensor = len(sensorIDs)
nTime = int(obsData.shape[0]/(frequency*interval))    # Num of 30 min intervals in this data file

# 10*10*4 m (x*y*z) domain, with grid spacing of (0.3*0.3*0.5 m), 可能的源范围
nx = hgt.shape[1]   # of grids in x direction
ny = hgt.shape[0]   # of grids in y direction

## Perferm source characterization model
postPDF = np.ones((nx, ny, nleak)) # posterior pdf
locationPDF = np.zeros((nx, ny, nTime)) # initial location pdf
strengthPDF = np.zeros((nleak, nTime))  # initial strength pdf

def get_wind_rose_obs_conc(wd, conc, nvalid=30, Thre=2.5):
    ''' 获取极坐标浓度 '''
    thetaDelta = 10/2 # degree
    thetas = np.arange(5,360,5)
    concMean, concStd, concDir = [], [], []

    for theta in thetas:
       indexs = np.logical_and(wd > theta-thetaDelta, wd < theta+thetaDelta)
       if sum(indexs) > nvalid:
           iConc = np.mean(conc[indexs])
           # print(theta, iConc)
           if iConc > Thre:
               concDir.append( theta )
               concStd.append( np.std( conc[indexs] ) )
               concMean.append( iConc )
    return concMean, concStd, concDir

def transformation(x0, y0, x1, y1, theta):
    ''' 坐标转换 '''
    rad = (180 + 90 - theta) / 180 * np.pi
    T = np.array([[np.cos(rad), np.sin(rad)], [- np.sin(rad), np.cos(rad)]])
    p = np.array([x1-x0, y1-y0])
    v = np.dot(T, p)
    return v[0], v[1]

for i in range(nTime):
   beg = int(i*frequency*interval)
   end = int(beg + frequency*interval)
   ustar = np.mean( obsData['speed_3D'].values[beg:end] )/10.
   for sensorID in sensorIDs: # 传感器
       sensorInfo = meta.loc[sensorID]
       # 求 每个方向上的浓度
       concObs, concStd, concDir = get_wind_rose_obs_conc(obsData['azimuth_3D'].values[beg:end], obsData[sensorID].values[beg:end])
       if len( concObs ) > 0:
           print(sensorID)
           for j in range(nx):
                for k in range(ny):
                    concMdl = np.zeros( (len(concObs)) )
                    for ii, iDir in enumerate(concDir):
                        sx = (j+1)*ds
                        sy = (k+1)*ds
                        thisX, thisY = transformation(sx, sy, sensorInfo.x, sensorInfo.y, iDir)
                        if thisX > 0: # downwind
                            # 插值得到一个结果, 忽略了源的高度; 源高度不同, 模拟的浓度不同, hgt?
                            xIndex = int( (thisX - xBeg)/dx -1 ) 
                            yIndex = int( (thisY - yBeg)/dy -1 ) 
                            zIndex = int( (sensorInfo.z - zBeg)/dz -1)
                            # 需要检查范围
                            concMdl[ii] = mdlData[zIndex, yIndex, xIndex]*0.5/ustar*SCFHToPPM

                    diff = np.zeros((nleak))
                    for jj in range(nleak):
                        # calculate the difference between expected and measured concentration
                        diff[jj] = np.mean(abs(concObs - concMdl*leakRates[jj])) # 浓度和排放之间是线性关系

                    sigmaE = max(concStd) + max(obsNoise, obsNoiseRatio*np.mean(concObs)) + mdlErrorRatio*np.mean(concMdl)*leakRates
                    likeP = np.exp(-0.5*(diff/sigmaE)**2)/sigmaE # 
                    postPDF[j, k, :] = postPDF[j, k, :]*likeP

   # Posterior divided by evidence: 为啥？
   postPDF = postPDF/np.sum(postPDF)

   # marginalize for source location (x,y)
   locationPDF[:, :, i] = np.sum(postPDF, axis=2)
   locationPDF[:, :, i] = locationPDF[:, :, i]/np.sum( locationPDF[:, :, i]) # sum = 1.

   # marginalize for source location (x,y)
   strengthPDF[:, i] = np.sum(postPDF, axis=(0, 1))
   strengthPDF[:, i] = strengthPDF[:, i]/np.sum( strengthPDF[:, i]) # sum = 1.

bestLeak = np.sum(strengthPDF[:, i]*leakRates)
bestLeakStd = np.sqrt(np.sum(strengthPDF[:, -1]*(leakRates-bestLeak)**2))

print('Estimated Leak Rate: ', bestLeak)
print('Leak rate Uncertainty: ',  bestLeakStd)

