"""Synthetic guard fixtures, never counted as scientific/model evidence."""
from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from ecodata.interface.cmaq_meteorology import apply


@pytest.mark.parametrize('fault',[None,'grid','nonfinite','fully_masked'])
def test_new_met_acceptance_rewrites_current_paths_and_rejects_bad_data(tmp_path,fault):
    case=tmp_path/'case';reference=case/'inputs/met/mcipv5.4/METCRO3D_12NE3_20180701.nc';reference.parent.mkdir(parents=True)
    met=tmp_path/'new_mcip';met.mkdir();folder=tmp_path/'run';folder.mkdir()
    attrs=dict(GDTYP=2,P_ALP=33.,P_BET=45.,P_GAM=-97.,XCENT=-97.,YCENT=40.,XORIG=1140000.,YORIG=-456000.,XCELL=12000.,YCELL=12000.,NCOLS=3,NROWS=2,NLAYS=2,VGTYP=7,VGTOP=5000.,VGLVLS=np.array([1.,.5,0.],dtype='f4'))
    for p in [reference,met/'METCRO3D.nc']:
        with Dataset(str(p),'w') as d:
            for n,size in [('TSTEP',2),('VAR',1),('DATE-TIME',2),('LAY',2),('ROW',2),('COL',3)]:d.createDimension(n,size)
            d.setncatts(attrs);d.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'))[:]=[[[2018182,0]],[[2018182,10000]]]
            d.createVariable('TA','f4',('TSTEP','LAY','ROW','COL'),fill_value=-9999.)[:]=280.
    with Dataset(str(met/'METCRO3D.nc'),'r+') as d:
        if fault=='grid':d.XORIG+=1.
        if fault=='nonfinite':d['TA'][0,0,0,0]=np.nan
        if fault=='fully_masked':d['TA'][:]=-9999.
    keys=['GRID_BDY_2D','GRID_CRO_2D','GRID_CRO_3D','GRID_DOT_2D','MET_CRO_2D','MET_CRO_3D','MET_DOT_3D','MET_BDY_3D','LUFRAC_CRO']
    (folder/'run.csh').write_text('setenv GRIDDESC old\nset METpath = old\n'+'\n'.join('setenv '+k+' old' for k in keys))
    if fault:
        with pytest.raises(ValueError):apply(folder,case,met)
        assert 'new_mcip' not in (folder/'run.csh').read_text()
    else:
        report=apply(folder,case,met);text=(folder/'run.csh').read_text()
        assert report['grid_and_time_verified'] and 'METCRO3D.nc' in text and ' old' not in text
