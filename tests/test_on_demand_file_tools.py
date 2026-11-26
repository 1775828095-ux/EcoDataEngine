"""Read-only tool boundaries, native streaming calls, and lazy GUI entry."""
from io import BytesIO
from pathlib import Path
import json
import zipfile
import pytest
from ecodata.desktop.file_tools import inspect
from ecodata.desktop.file_agent import FileTools, scan_index
from ecodata.desktop.deepseek_transport import read_reply


def bundle(tmp_path):
    inner=BytesIO()
    with zipfile.ZipFile(inner,'w') as z:
        z.writestr('config/file.cio','weather_file = climate.pcp\n')
        z.writestr('config/climate.pcp','year,day,rain\n2020,1,3\n')
    target=tmp_path/'source.zip'
    with zipfile.ZipFile(target,'w') as z:
        z.writestr('nested.zip',inner.getvalue())
        z.writestr('unread.bin',b'x'*100000)
    return target


def test_listing_does_not_decode_unrequested_members(tmp_path):
    root=bundle(tmp_path)
    result=inspect(root,'list_directory',dict(path=''))
    assert len(result['entries'])==2 and result['arrays_read'] is False
    assert result['entries'][0]['path']=='nested.zip'
    scan,_=scan_index(root)
    assert scan['on_demand'] and scan['rows']==[]


def test_nested_config_header_and_real_refs(tmp_path):
    root=bundle(tmp_path);tools=FileTools(root)
    listing=tools.call('list_directory',dict(path='nested.zip'))
    assert listing['ok']
    result=tools.call('read_text',dict(path='nested.zip!/config/file.cio'))
    assert 'climate.pcp' in result['result']['text']
    assert 'ref' not in result['result']['row']
    row=tools.rows[result['result']['row']['name']]
    from ecodata.desktop.scan import local_member
    with local_member(row) as path:assert path.read_text().startswith('weather_file')
    assert all(e['temporary_directory_removed'] for e in tools.events)
    assert len(tools.rows)==1


def test_paths_unknown_tools_and_input_change(tmp_path):
    root=bundle(tmp_path);tools=FileTools(root)
    with pytest.raises(ValueError):inspect(tmp_path,'read_text',dict(path='../outside'))
    with pytest.raises(ValueError):inspect(root,'read_text',dict(path='../outside'))
    with pytest.raises(ValueError):tools.call('run_shell',dict(path=''))
    root.write_bytes(b'changed')
    with pytest.raises(ValueError,match='变化'):tools.call('list_directory',dict(path=''))


def test_input_changed_during_read_is_not_used(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import subprocess
    root=bundle(tmp_path);tools=FileTools(root)
    def change(*args,**kwargs):
        root.write_bytes(b'changed while worker reads')
        return SimpleNamespace(returncode=0,stdout=json.dumps(dict(ok=True,result={})))
    monkeypatch.setattr(subprocess,'run',change)
    with pytest.raises(ValueError,match='读取期间'):tools.call('list_directory',dict(path=''))


def test_native_streaming_tool_calls_are_reassembled():
    parts=[dict(id='call1',type='function',index=0,function=dict(name='read_text',arguments='{"path":')),
           dict(index=0,function=dict(arguments='"README.md"}'))]
    chunks=[dict(choices=[dict(index=0,delta=dict(tool_calls=[part]),finish_reason=None)]) for part in parts]
    chunks.append(dict(choices=[dict(index=0,delta={},finish_reason='tool_calls')]))
    stream=BytesIO((''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode())
    reply=read_reply(stream,lambda _:None)
    call=reply['choices'][0]['message']['tool_calls'][0]
    assert call['id']=='call1' and json.loads(call['function']['arguments'])==dict(path='README.md')


def test_selected_netcdf_metadata(tmp_path):
    import xarray as xr
    p=tmp_path/'field.nc'
    xr.Dataset({'rain':(('y','x'),[[1.,2.],[3.,4.]],dict(units='mm'))}).to_netcdf(p,engine='scipy')
    result=inspect(p,'read_header',dict(path=''))
    assert any(f['name']=='rain' and f['unit']=='mm' for f in result['row']['fields'])


def test_shared_entry_uses_only_model_requested_files(tmp_path,monkeypatch):
    from ecodata.desktop import service
    from ecodata.desktop.discovery import discover
    from ecodata.desktop.three_phase import run
    from ecodata.desktop.scan import Scanner
    root=bundle(tmp_path);scan,_=scan_index(root);summary,_=discover(scan,'swat_plus')
    summary.update(task_goal='evaluate_gain',task_requirements=dict(time_investigation='check original receipt'))
    monkeypatch.setattr(Scanner,'scan',lambda *a,**k:pytest.fail('whole-input scan must not run'))
    turns=iter([
        dict(role='assistant',content=None,tool_calls=[dict(id='one',type='function',
            function=dict(name='list_directory',arguments=json.dumps(dict(path='nested.zip'))))]),
        dict(role='assistant',content=None,tool_calls=[dict(id='two',type='function',
            function=dict(name='read_text',arguments=json.dumps(dict(path='nested.zip!/config/file.cio'))))]),
        dict(role='assistant',content=json.dumps(dict(summary='read config',engineering_evidence=[],dependencies=[],unknowns=[])))])
    def tool_turn(messages, tools, api_key, progress):
        assert api_key == 'unit-test'
        return next(turns), dict(model='unit-test')
    monkeypatch.setattr(service,'tool_turn',tool_turn)
    def decide(request,*args,**kwargs):
        assert kwargs['api_key']=='unit-test'
        contents=json.loads(request.summary)
        if request.stage == "file_classification":
            return dict(classifications=[dict(file=file["id"], kind="uncertain") for file in contents["files"]])
        assert contents['task_goal']=='evaluate_gain'
        assert contents['task_requirements']['time_investigation']=='check original receipt'
        return dict(decisions=[],questions=[])
    monkeypatch.setattr(service,'send',decide)
    result=run(scan,summary,{},False,'unit-test',lambda _:None,lambda _:None)
    full=result[-1]
    assert len(full['rows'])==1 and full['rows'][0]['name'].endswith('file.cio')
    assert full['file_agent_audit']['tool_calls']==2
    assert all(e['temporary_directory_removed'] for e in full['file_agent_audit']['events'])


def test_gui_scan_entry_keeps_directory_only(tmp_path,monkeypatch):
    import os,time
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from ecodata.desktop.__main__ import Window
    root=bundle(tmp_path);app=QApplication.instance() or QApplication([])
    window=Window();window.path.setText(str(root));window.scan_now()
    deadline=time.monotonic()+15
    while window.thread is not None and time.monotonic()<deadline:
        app.processEvents();time.sleep(.02)
    assert window.scan and window.scan['on_demand'] and window.scan['rows']==[]
    window.batch_timer.stop()
    window.close();app.processEvents()


def test_local_file_cards_precede_targeted_deepseek_read(tmp_path, monkeypatch):
    from ecodata.desktop import discovery, service
    from ecodata.desktop.file_agent import run

    root = tmp_path / "input.zip"
    with zipfile.ZipFile(root, "w") as archive:
        archive.writestr("docs/README.md", "# Project setup\nPRIVATE_DESCRIPTION\n")
        archive.writestr("data/rain.csv", "station,rain\nPRIVATE_STATION,2.4\nOTHER_STATION,3.0\n")
        for number in range(65):
            archive.writestr(f"other/note-{number}.bin", b"opaque")
    scan, _ = scan_index(root)
    assert len(scan["file_cards"]) == 67 and scan["rows"] == []
    summary, _ = discovery.discover(scan, "data_only")
    classified = []

    def send(request, *args, **kwargs):
        if request.stage != "file_classification":
            return dict(decisions=[], questions=[])
        files = json.loads(request.summary)["files"]
        classified.extend(files)
        assert all("kind" not in file for file in files)
        assert "PRIVATE_" not in request.summary
        # An intentionally different judgement verifies the tool context uses the model's reply.
        return dict(classifications=[dict(file=file["id"], kind={"md": "mixed", "CSV": "data"}.get(
            file["format"], "uncertain")) for file in files])

    def turn(messages, tools, key, progress):
        if len(messages) == 2:
            assert len(classified) == 67
            first = json.loads(messages[1]["content"])["file_types"]
            cards = {item["path"]: item for item in first["entries"]}
            assert first["total"] == 67 and first["next_offset"] == 60
            assert cards["docs/README.md"]["basis"] == "heading_prefix"
            assert cards["docs/README.md"]["kind"] == "mixed"
            assert cards["data/rain.csv"]["kind"] == "data" and cards["data/rain.csv"]["columns"] == 2
            assert "PRIVATE_" not in messages[1]["content"]
            return dict(role="assistant", tool_calls=[
                dict(id="one", function=dict(name="list_file_types", arguments=json.dumps(dict(path="", offset=60)))),
                dict(id="two", function=dict(name="read_header", arguments=json.dumps(dict(path="data/rain.csv")))),
            ]), {}
        remaining = json.loads(messages[-2]["content"])["result"]
        assert len(remaining["entries"]) == 7 and remaining["next_offset"] is None
        header = json.loads(messages[-1]["content"])["result"]
        assert header["header_lines"] == ["station,rain"]
        return dict(role="assistant", content='{"summary":"字段已核对"}'), {}

    monkeypatch.setattr(service, "tool_turn", turn)
    monkeypatch.setattr(service, "send", send)
    *_, full = run(scan, summary, {}, False, "synthetic", lambda _: None, lambda _: None)
    assert [row["name"].split("!/")[-1] for row in full["rows"]] == ["data/rain.csv"]


def test_declared_goal_is_not_erased_by_missing_validation(tmp_path):
    import numpy as np
    import xarray as xr
    from ecodata.desktop import discovery,catalog_decision
    from ecodata.desktop.scan import Scanner
    for number,step in enumerate([1.,.5]):
        axis=np.arange(1.,3.01,step)
        xr.Dataset({'temperature':(('time','lat','lon'),np.full((2,len(axis),len(axis)),280.+number),
            dict(units='K',physical_variable='air_temperature',temporal_semantics='instantaneous',spatial_semantics='grid_estimate'))},
            coords=dict(time=np.array(['2020-01-01','2020-01-02'],dtype='datetime64[ns]'),lat=axis,lon=axis),
            attrs=dict(crs='EPSG:4326')).to_netcdf(tmp_path/f'source{number}.nc',engine='scipy')
    scan,_=Scanner().scan(tmp_path)
    summary,_=discovery.discover(scan,'swat_plus')
    originals,details=catalog_decision.local_fields(scan,summary)
    fields=[value for value in originals.values() if value[2]['name']=='temperature']
    pair,local=discovery.describe_pair(scan,'C1',*fields,'swat_plus',details)
    reply=dict(decisions=[dict(candidate='C1',action='pending_goal',goal='unknown',reason='missing validation evidence')])
    for goal,expected in [('auto',False),('evaluate_gain',True)]:
        _,gated=discovery.check(scan,{'C1':local},reply,dict(summary,candidates=[pair],task_goal=goal))
        assert (('C1','fusion') in gated)==expected
        assert ('C1','align_space') in gated


@pytest.mark.parametrize('second_tool',[False,True])
def test_final_json_gets_only_one_bounded_format_correction(tmp_path,monkeypatch,second_tool):
    from ecodata.desktop import service,discovery
    from ecodata.desktop.file_agent import run,scan_index
    root=bundle(tmp_path);scan,_=scan_index(root);summary,_=discovery.discover(scan,'swat_plus')
    messages=[]
    def turn(history,*args):
        messages.append(list(history))
        if second_tool and len(messages)==2:
            return dict(role='assistant',tool_calls=[dict(id='not-executed',function=dict(name='read_header',arguments='{}'))]),dict(model='functional-test')
        return dict(role='assistant',content='{broken'),dict(model='functional-test')
    monkeypatch.setattr(service,'tool_turn',turn)
    monkeypatch.setattr(service, "send", lambda request, *args, **kwargs: dict(classifications=[
        dict(file=file["id"], kind="uncertain") for file in json.loads(request.summary)["files"]]))
    with pytest.raises(ValueError,match='连续两次|修正阶段'):
        run(scan,summary,{},False,'test-key',lambda _:None,lambda _:None)
    assert len(messages)==2
    assert '不重新读取文件' in messages[-1][-1]['content']
