import numpy as np 
#import pyfits
import astropy.io.fits as pyfits
import os
import datetime
import time
import sys
from decimal import Decimal
import scipy.stats as ss
secperday = 3600 * 24

filename = sys.argv[1]

hdulist = pyfits.open(filename)
#print len(hdulist) 
hdu0 = hdulist[0]
hdu1 = hdulist[1]
data1 = hdu1.data['data']
header1 = hdu1.header
print(header1['NBITS'])
print(data1.shape)
print(data1.dtype)
#print hdu0.header
#print hdu1.header

print(data1.shape)
m,n,p,c,x = data1.shape
data = data1.squeeze()
print(data.shape)

size = data.flatten().size
rawdata = data.flatten().reshape((size,1))
onebit = np.unpackbits(rawdata, axis=1)[...,::-1]
newdata = np.packbits(onebit.flatten().reshape((-1,2)), axis=1)


#data = data.sum(axis=-1)
#print data.shape
data = newdata.reshape((-1, p, c))


#print data.shape
#data = data.reshape((4096, 64, p,4096)).sum(axis=2)
#print data.shape
data = data.sum(axis=1).T
print(data.shape)


l, m = data.shape
data = data.reshape(l, int(m/128), 128).sum(axis=2)
#data = data.reshape( (l/16, 16,  m/64) ).sum(axis=1)
#data = data.reshape( (l, m/1, 1) ).sum(axis=2)
print(data.shape)

#data = (data - data.mean(axis=0))

#import matplotlib
#matplotlib.use('Agg') 
from matplotlib import pyplot as plt 
fig = plt.figure()

#imshow(data[2800:,25:], aspect='auto', origin='bottomleft')
# plt.imshow(data[:,:], aspect='auto', origin='bottomleft')
# plt.imshow(data[:,:], aspect='auto', origin='lower')
plt.imshow(data, aspect='auto', origin='lower', cmap=plt.cm.gray)
plt.colorbar()
plt.show()
#plt.savefig(rootname+'.png')
print("Done")
'''

plt.plot(data[:,:].mean(axis=1))#, origin='bottomleft')
plt.show()

plt.plot(data.std(axis=1))
plt.show()

plt.plot(ss.kurtosis(data, axis=1))
plt.show()
#plt.savefig(rootname+'_bp.png')
#show()

'''
