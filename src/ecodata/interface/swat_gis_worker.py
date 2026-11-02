"""Isolated official QSWAT+ 4.0.3 adapter; run under QGIS Python only.

This worker builds GIS objects, not simulation results. A fresh output directory
is mandatory. No completed reference project is accepted or copied.
"""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys


def require_delineation_output(path, snap_status, diagnostics=None):
    """QSWAT reports some failures without raising; never continue with no basin."""
    if not path or not Path(path).is_file():
        detail = ''
        if diagnostics and diagnostics.get('points'):
            detail = '；各出口到主河网最短距离（米）：' + ', '.join(
                str(p['id']) + '=' + format(p['stream_distance_m'], '.3f')
                for p in diagnostics['points'] if p.get('stream_distance_m') is not None)
        raise RuntimeError(
            '官方流域划分未生成有效流域栅格；出口吸附状态：' + str(snap_status)
            + detail
            + '。请核对出口位置、河网阈值及允许吸附距离，详见划分日志；'
              '平台不会自动移动出口或放宽参数。')


def outlet_diagnostics(outlets, networks, stream_cells, channel_cells, snap_metres):
    """Read-only distances to generated networks; never relocate an outlet."""
    from osgeo import ogr, osr
    result = dict(stream_cells=stream_cells, channel_cells=channel_cells,
                  snap_metres=snap_metres, points=[], errors=[])
    source = ogr.Open(str(outlets))
    if source is None:
        result['errors'].append('无法读取出口图层')
        return result
    layer = source.GetLayer()
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            result['errors'].append('出口几何为空：' + str(feature.GetFID()))
            continue
        point = dict(id=feature.GetFID())
        for kind in ('stream', 'channel'):
            path = networks.get(kind, '')
            network = ogr.Open(str(path)) if path else None
            if network is None:
                point[kind + '_distance_m'] = None
                continue
            reaches = network.GetLayer()
            origin, target = layer.GetSpatialRef(), reaches.GetSpatialRef()
            if origin is None or target is None or not target.IsProjected() or abs(target.GetLinearUnits()-1) > 1e-9:
                result['errors'].append('出口/河网缺少可核验米制坐标')
                continue
            origin, target = origin.Clone(), target.Clone()
            origin.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transformed = geometry.Clone()
            if not origin.IsSame(target):
                if transformed.Transform(osr.CoordinateTransformation(origin,target)) != 0:
                    result['errors'].append('出口坐标转换失败')
                    continue
            distances = [transformed.Distance(f.GetGeometryRef()) for f in reaches if f.GetGeometryRef() is not None]
            point[kind + '_distance_m'] = min(distances) if distances else None
        result['points'].append(point)
    return result


def build(config_file):
    config = json.loads(Path(config_file).read_text(encoding='utf-8-sig'))
    required = ('output', 'support_dir', 'plugin_dir', 'dem', 'landuse', 'soil',
                'outlets', 'landuse_lookup', 'soil_lookup', 'usersoil',
                'stream_cells', 'channel_cells', 'snap_metres', 'slope_limits')
    for key in required:
        if key not in config:
            raise ValueError('缺少建工程设置：' + key)
    output = Path(config['output'])
    if output.exists():
        raise ValueError('必须使用新的工作目录，不能覆盖已有工程')
    for key in ('dem', 'landuse', 'soil', 'outlets', 'landuse_lookup', 'soil_lookup', 'usersoil'):
        if not Path(config[key]).exists():
            raise ValueError('原始资料不存在：' + key)
    for key in ('stream_cells', 'channel_cells', 'snap_metres'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(key + '必须为正整数')
    limits = config['slope_limits']
    if not isinstance(limits, list) or any(not isinstance(x, (int, float)) or not 0 < x < 1000 for x in limits) or limits != sorted(set(limits)):
        raise ValueError('坡度分级必须为递增、不重复的正数列表（百分比）')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from qgis.core import QgsApplication, QgsProject
    from qgis.PyQt.QtCore import QSettings, QCoreApplication, QFileInfo
    from osgeo import gdal
    app = QgsApplication([], False)
    app.initQgis()
    sys.path.insert(0, str(Path(os.environ['QGIS_PREFIX_PATH']) / 'python/plugins'))
    sys.path.insert(0, config['plugin_dir'])
    from processing.core.Processing import Processing
    Processing.initialize()
    from QSWATPlus.QSWATPlusMain import QSWATPlus
    from QSWATPlus.delineation import Delineation
    from QSWATPlus.hrus import HRUs
    from QSWATPlus.parameters import Parameters
    from QSWATPlus.QSWATUtils import QSWATUtils, FileTypes
    # Official Windows batch mode falls back to this constant even when a
    # SWATPlusDir setting exists; configure this process, never edit the plugin.
    Parameters._SWATPLUSDEFAULTDIR = config['support_dir']
    output.mkdir(parents=True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / 'settings'))
    QCoreApplication.setOrganizationName('EcoDataEngine')
    QCoreApplication.setApplicationName('SWATBuild')
    QSettings().setValue('/QSWATPlus/SWATPlusDir', config['support_dir'])

    class Interface:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self
        def layers(self):
            return list(QgsProject.instance().mapLayers().values())

    project = QgsProject.instance()
    project.setTitle('Watershed')
    project.setFileName(str(output / 'Watershed.qgs'))
    project.write()
    plugin = QSWATPlus(Interface())
    plugin.setupProject(project, True)
    gv = plugin._gv
    root = project.layerTreeRoot()
    print('STAGE: delineation', flush=True)
    delin = Delineation(gv, plugin._demIsProcessed)
    delin.init()
    delin._dlg.numProcesses.setValue(0)

    def raster(key, directory):
        src = gdal.Open(config[key])
        if src is None:
            raise ValueError('无法读取栅格：' + key)
        dest = str(Path(directory) / (key + '.tif'))
        copied = gdal.GetDriverByName('GTiff').CreateCopy(dest, src)
        if copied is None:
            raise ValueError('无法转换栅格：' + key)
        copied.FlushCache()
        copied = src = None
        return dest

    dem = raster('dem', gv.demDir)
    delin._dlg.selectDem.setText(dem)
    layer, _ = QSWATUtils.getLayerByFilename(root.findLayers(), dem, FileTypes._DEM, gv, None, QSWATUtils._WATERSHED_GROUP_NAME)
    if layer is None or layer.crs().isGeographic() or layer.crs().mapUnits().name != 'Meters':
        raise ValueError('当前建工程要求有明确米制投影的DEM；不猜测投影')
    delin.setDefaultNumCells(layer)
    delin._dlg.numCellsSt.setText(str(config['stream_cells']))
    delin._dlg.numCellsCh.setText(str(config['channel_cells']))
    QSWATUtils.copyFiles(QFileInfo(config['outlets']), gv.shapesDir)
    delin._dlg.selectOutlets.setText(str(Path(gv.shapesDir) / Path(config['outlets']).name))
    delin._dlg.useOutlets.setChecked(True)
    delin._dlg.snapThreshold.setText(str(config['snap_metres']))
    if config.get('burn_streams'):
        from osgeo import ogr
        streams=ogr.Open(config['burn_streams'])
        if streams is None or streams.GetLayer().GetFeatureCount()<=0:
            raise ValueError('刻蚀河网为空或不可读')
        lines=streams.GetLayer()
        if ogr.GT_Flatten(lines.GetGeomType()) not in (ogr.wkbLineString,ogr.wkbMultiLineString):
            raise ValueError('刻蚀河网必须为线图层')
        reference=lines.GetSpatialRef()
        if reference is None or not reference.IsProjected() or abs(reference.GetLinearUnits()-1)>1e-9:
            raise ValueError('刻蚀河网需要明确米制投影，不猜测坐标')
        dem_reference=gdal.Open(dem).GetSpatialRef()
        if dem_reference is None or not reference.IsSame(dem_reference):
            raise ValueError('刻蚀河网与DEM坐标系不一致，请先明确转换，不隐式重投影')
        streams=None
        depth=config.get('burnin_depth_metres')
        if type(depth) is not int or not 0<depth<=1000:raise ValueError('需明确河网刻蚀深度（米）')
        gv.burninDepth=depth
        delin._dlg.checkBurn.setChecked(True)
        delin._dlg.selectBurn.setText(config['burn_streams'])
        gv.burnFile=config['burn_streams']
        print('EXPLICIT STREAM BURN: '+config['burn_streams']+'; depth_metres='+str(depth),flush=True)
    delin.runTauDEM2()
    diagnostics = outlet_diagnostics(config['outlets'], {'stream': gv.streamFile, 'channel': gv.channelFile}, config['stream_cells'],
                                     config['channel_cells'], config['snap_metres'])
    (output / 'outlet_diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    require_delineation_output(gv.channelBasinFile, delin._dlg.snappedLabel.text(), diagnostics)
    delin.finishDelineation()
    if not plugin._odlg.hrusButton.isEnabled():
        raise RuntimeError('官方流域划分未完成，请检查TauDEM日志及出口位置')
    # Check required categories on the actual delineated watershed, not the
    # rectangular DEM extent. Nearest-neighbour sampling preserves categories;
    # outside-basin NoData is a normal mask, never filled.
    import numpy as np
    import csv
    basin = gdal.Open(gv.channelBasinFile)
    band = basin.GetRasterBand(1)
    basin_values = band.ReadAsArray()
    # TauDEM channel IDs are zero-based: basin 0 is valid, not a missing cell.
    active = np.isfinite(basin_values) & (basin_values != band.GetNoDataValue())
    gt = basin.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0 or not active.any():
        raise ValueError('无法核验流域有效区；旋转网格暂未接入')
    bounds = (gt[0], gt[3] + gt[5]*basin.RasterYSize,
              gt[0] + gt[1]*basin.RasterXSize, gt[3])
    checks = {}
    for key in ('landuse', 'soil'):
        sampled = gdal.Warp('', config[key], format='MEM', outputBounds=bounds,
                            width=basin.RasterXSize, height=basin.RasterYSize,
                            dstSRS=basin.GetProjection(), resampleAlg='near',
                            outputType=gdal.GDT_Float64, dstNodata=-1e30)
        if sampled is None:
            raise ValueError('无法对齐分类栅格：' + key)
        values = sampled.ReadAsArray()[active]
        if not np.isfinite(values).all() or (values == -1e30).any() or (values != np.floor(values)).any():
            raise ValueError(key + '在流域有效区存在缺失或非整数分类，不填补')
        with open(config[key + '_lookup'], encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))[1:]
        ids = {int(r[0]) for r in rows}
        actual = set(map(int, np.unique(values)))
        if not actual <= ids:
            raise ValueError(key + '缺少分类映射：' + str(sorted(actual - ids)))
        checks[key] = {'active_cells': int(active.sum()), 'categories': sorted(actual), 'missing': 0}
    (output / 'active_grid_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('STAGE: HRU', flush=True)
    hrus = HRUs(gv, plugin._odlg.reportsBox)
    hrus.init()
    db = gv.db
    db.plantSoilDatabase = db.dbRefFile
    db.plantSoilDatabaseSelected = True
    for typ in ('plant', 'urban'):
        if config.get(typ):
            if not db.importCsv(typ, typ, config[typ]):
                raise ValueError('参数导入失败：' + typ)
    for key, typ, name in (('landuse_lookup', 'landuse', 'input_landuse'),
                            ('soil_lookup', 'soil', 'input_soil'),
                            ('usersoil', 'usersoil', 'input_usersoil')):
        if not db.importCsv(name, typ, config[key]):
            raise ValueError('参数导入失败：' + key)
    db.usersoilTable = 'input_usersoil'
    db.slopeLimits = limits
    hrus.landuseTable = 'input_landuse'
    hrus.soilTable = 'input_soil'
    for key, filetype, group in (('landuse', FileTypes._LANDUSES, QSWATUtils._LANDUSE_GROUP_NAME),
                                  ('soil', FileTypes._SOILS, QSWATUtils._SOIL_GROUP_NAME)):
        path = raster(key, gv.landuseDir if key == 'landuse' else gv.soilDir)
        layer, _ = QSWATUtils.getLayerByFilename(root.findLayers(), path, filetype, gv, None, group)
        setattr(hrus, key + 'File', path)
        setattr(hrus, key + 'Layer', layer)
    hrus._dlg.channelMergeVal.setText('0')
    hrus._dlg.generateFullHRUs.setChecked(True)
    if not hrus.readFiles():
        raise RuntimeError('官方HRU资料读取失败')
    hrus._dlg.filterLanduseButton.setChecked(True)
    hrus.CreateHRUs.landuseVal = 0
    hrus.CreateHRUs.soilVal = 0
    hrus.CreateHRUs.slopeVal = 0
    hrus.calcHRUs()
    if not hrus.completed:
        raise RuntimeError('官方HRU生成未完成')
    project.write()
    with sqlite3.connect(db.dbFile) as conn:
        counts = {t: conn.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                  for t in ('gis_hrus', 'gis_channels', 'gis_subbasins')}
    if any(n <= 0 for n in counts.values()):
        raise RuntimeError('生成工程的GIS对象为空')
    result = {'stage': 'gis_only', 'project_db': db.dbFile, 'datasets_db': db.dbRefFile,
              'counts': counts, 'model_run': False}
    (output / 'gis_build_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    try:
        build(sys.argv[1])
    except Exception as exc:
        # Keep traceback in build.log and a concise, structured reason for UI.
        Path(sys.argv[1]).with_name('worker_error.json').write_text(
            json.dumps({'error': str(exc)}, ensure_ascii=False), encoding='utf-8')
        raise
