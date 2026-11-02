"""Stream complete native WPS fields; subset original cells without interpolation."""
import struct
import numpy as np
from ecodata.interface.wrf_assembly import read_fields

def stream_fields(source):
    with source.open('rb') as src:
        while True:
            pieces=[]
            for i in range(5):
                head=src.read(4)
                if not head and i==0:return
                if len(head)!=4:raise ValueError('Truncated original WPS marker')
                size=struct.unpack('>i',head)[0]
                if not 0<size<=40_000_000:raise ValueError('Original WPS field record exceeds bound')
                block=src.read(size);tail=src.read(4)
                if len(block)!=size or tail!=head:raise ValueError('Truncated or inconsistent original WPS record')
                pieces.extend((head,block,tail))
            yield read_fields(b''.join(pieces))[0]

def crop(source,target,latitude,longitude):
    count=0
    with target.open('wb') as dst:
        for f in stream_fields(source):
            lat,lon,dy,dx,r=f.grid
            iy=[i for i in range(f.ny) if abs(lat+i*dy-latitude)<=10]
            ix=[i for i in range(f.nx) if abs(((lon+i*dx-longitude+180)%360)-180)<=12]
            if len(ix)<2 or len(iy)<2 or ix!=list(range(ix[0],ix[-1]+1)):
                raise ValueError('Source coverage insufficient or longitude seam unsupported')
            p=list(f.parts);h=bytearray(p[1]);struct.pack_into('>ii',h,144,len(ix),len(iy));p[1]=bytes(h)
            p[2]=p[2][:8]+struct.pack('>5f',lat+iy[0]*dy,lon+ix[0]*dx,dy,dx,r)
            a=np.frombuffer(p[4],dtype='>f4').reshape(f.ny,f.nx);p[4]=a[np.ix_(iy,ix)].astype('>f4').tobytes()
            for block in p:
                size=struct.pack('>i',len(block));dst.write(size);dst.write(block);dst.write(size)
            count+=1
    if not count:raise ValueError('Empty original WPS file')
    return count
