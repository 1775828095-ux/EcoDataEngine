"""Pinned official SWAT+ Editor calls in an isolated process; no model solver."""
import json
import os
import sqlite3
import sys
from pathlib import Path


def export(job):
    sys.path.insert(0, job['api'])
    from actions.setup_project import SetupProject
    from actions.import_weather import Swat2012WeatherImport, WeatherImport, WgnImport
    from actions.write_files import WriteFiles
    from database.project.config import Project_config
    from database.project.setup import SetupProjectDatabase

    path = Path(job['project'])
    if job['initialise']:
        SetupProject(str(path), '4.0.0', project_name=job['project_name'], datasets_db=job['datasets'])
    SetupProjectDatabase.init(str(path))
    output = path.parent / 'Scenarios' / 'Default' / 'TxtInOut'
    weather = path.parent / '_ecodata_weather'
    weather.mkdir(exist_ok=True)
    Project_config.update(input_files_dir=str(output), weather_data_dir=str(weather),
                          weather_data_format='plus', project_directory=str(path.parent),
                          project_db=path.name, reference_db=job['datasets'],
                          wgn_db=job.get('wgn'),wgn_table_name=job.get('wgn_table')).execute()
    if job['weather_format'] == '2012':
        Swat2012WeatherImport(str(path), False, True, job['weather_source']).import_data()
    else:
        Project_config.update(weather_data_dir=job['weather_source']).execute()
        WeatherImport(str(path), False, True).import_data()
    if job.get('wgn'):
        WgnImport(str(path),False,False,'database').import_data()
    elif job.get('wgn_csv'):
        WgnImport(str(path),False,False,'two_file',*job['wgn_csv']).import_data()
    # Validate database filenames before allowing the official text writer to
    # create files. No uploaded file is executed or imported as Python.
    db = sqlite3.connect(path)
    for table, column in [('file_cio', 'file_name'), ('weather_file', 'filename')]:
        columns = {r[1] for r in db.execute('pragma table_info('+table+')')}
        if column not in columns:
            continue
        for (name,) in db.execute('select '+column+' from '+table):
            if name and name != 'null' and (Path(name).name != name or ':' in name or '\\' in name):
                raise ValueError('Unsafe generated filename: '+str(name))
    db.close()
    WriteFiles(str(path), '62.0.0').write()
    if not (output/'file.cio').is_file():
        raise ValueError('Official writer did not generate file.cio')
    Project_config.update(project_directory='',project_db=path.name,
        reference_db=os.path.relpath(job['datasets'],path.parent),
        wgn_db=os.path.relpath(job['wgn'],path.parent) if job.get('wgn') else None,
        input_files_dir='Scenarios/Default/TxtInOut',
        weather_data_dir=os.path.relpath(job['weather_source'] if job['weather_format']=='plus' else weather,path.parent)).execute()
    print(json.dumps(dict(status='EXPORTED', output=str(output), solver_executed=False)))


if __name__ == '__main__':
    export(json.loads(Path(sys.argv[1]).read_text(encoding='utf8')))
