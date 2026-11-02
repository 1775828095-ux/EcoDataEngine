"""Mixed gridded/discrete records follow native WROUT1 and OUTPUT order."""
import struct
import numpy as np
import pytest
from ecodata.interface.calpuff_results import read_concentrations
from ecodata.core.exceptions import SchemaError


def fixture(bad_count=False):
    header=bytearray(320);header[:12]=b'CALPUFF'.ljust(12)
    integers=np.zeros(47,dtype='<i4')
    integers[0:3]=[1990,9,4];integers[5]=1;integers[8:10]=2
    integers[20:25]=[1,1,2,2,1];integers[33:39]=[0,2,0,1,1,1]
    floats=integers.view('<f4');floats[4]=5;floats[10:12]=1;floats[13:15]=[310,4820]
    header[36:224]=integers.tobytes()
    label=b'SO2'.ljust(12)+b'  1'
    coords=np.array([[310.5,311.5],[4820.5,4821.5],[20,30]],dtype='<f4')
    chunks=[b'CONC.DAT'.ljust(16)+b'2.1'.ljust(16)+bytes(64),struct.pack('<i',0),bytes(header),b'ignored species definitions',label,coords.tobytes(),struct.pack('<8i',1990,9,4,0,1990,9,5,0),bytes(32),struct.pack('<i',3),label+np.array([-2,1,2],dtype='<f4').tobytes(),struct.pack('<i',3 if bad_count else 2),label+np.array([5,6],dtype='<f4').tobytes()]
    return b''.join(struct.pack('<i',len(c))+c+struct.pack('<i',len(c)) for c in chunks)


def test_grid_and_discrete_are_not_confused():
    result=read_concentrations(fixture())
    np.testing.assert_array_equal(result['values_g_m3']['SO2'],[[[0,0],[1,2]]])
    np.testing.assert_array_equal(result['discrete_values_g_m3']['SO2'],[[5,6]])
    np.testing.assert_array_equal(result['discrete_coordinates_km_m'][2],[20,30])


def test_invalid_discrete_compression_rejected():
    with pytest.raises(SchemaError,match='压缩计数越界'):read_concentrations(fixture(True))
