"""Repeatable desktop acceptance, using bundled data and a synthetic held-out fixture."""
import argparse
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import threading

import psutil
from PySide6.QtWidgets import QApplication
from ecodata.desktop.__main__ import Window
from ecodata.desktop.scan import Scanner
from ecodata.desktop import service as s

ROOT=Path(__file__).resolve().parents[1]

def benchmark(scanner,path):
    process=psutil.Process()
    initial=process.memory_info().rss
    peak=[initial]
    stop=threading.Event()
    def measure():
        while not stop.wait(.001):
            peak[0]=max(peak[0],process.memory_info().rss)
    t=threading.Thread(target=measure)
    t.start()
    try: report,metrics=scanner.scan(path)
    finally: stop.set();t.join()
    peak[0]=max(peak[0],process.memory_info().rss)
    return dict(metrics,initial_working_set_bytes=initial,peak_working_set_bytes=peak[0],
                sampled_working_set_delta_bytes=peak[0]-initial,member_count=len(report["rows"]), warnings=report["warnings"])

def approve(brief,folder,tag):
    p=s.preview(brief,folder/f"{tag}.json")
    (folder/f"{tag}-request.json").write_bytes(p.body)
    return s.send(p,p.digest,mock=True)

def acceptance(out,source):
    # Launch and exercise native widgets. No network or GUI screenshot upload.
    app=QApplication([])
    window=Window()
    window.show()
    app.processEvents()
    window.path.setText(str(source))
    window.scan_now()
    from PySide6.QtCore import QEventLoop,QTimer
    loop=QEventLoop()
    timer=QTimer()
    timer.timeout.connect(lambda:loop.quit() if window.thread is None else None)
    timer.start(20)
    QTimer.singleShot(120000,loop.quit)
    loop.exec()
    timer.stop()
    assert window.scan and window.table.rowCount()>0
    records=dict(native_window="已启动并通过 Qt 后台任务显示轻扫",scan_source=source.name,
                 scans=[],api="模拟；真实 API 未验证")
    scanner=Scanner()
    records["scans"]=[benchmark(scanner,source),benchmark(scanner,source)]
    window.close()
    (out/"desktop-measurements.json").write_text(s.dump(records),encoding="utf8")
    with TemporaryDirectory(prefix="eco-desktop-test-") as tmp:
        stage=Path(tmp)
        import runpy
        items,request,_=runpy.run_path(str(ROOT/"tests/test_spatial_relationship_fusion.py"))["fixture"]()
        fusion=stage/"fusion"
        fusion.mkdir()
        for item in items: (fusion/item.filename).write_bytes(item.content)
        (fusion/"README.md").write_text("grid0.nc is the background field.\nobs.csv provides station observations.\ntarget.nc declares target coordinates.",encoding="utf8")
        scan,_=Scanner().scan(fusion)
        source_roles={i.filename:i.role for i in items}
        sel=[(i,source_roles[r["name"]]) for i,r in enumerate(scan["rows"]) if r["name"] in source_roles]
        assert {"observations","background","target"}<={role for r in scan["rows"] for role in r["roles"]}
        state,local=s.prepare(scan,sel,"fusion",None,"","NetCDF","temperature","K","instantaneous","{}")
        plan=approve(s.planning_summary(scan,sel,"fusion","NetCDF"),out,"fusion-plan")
        state["purpose"]="验证站点观测能否校正背景场"
        result=s.run(state,stage/"fusion-job",plan["algorithms"])
        review=approve(s.review_summary(result),out,"fusion-review")
        (out/"fusion.zip").write_bytes(s.bundle(result,plan,review))
        records["fusion"]=dict(local=local["status"],confirmed=result["search"]["confirmed_initial"],
                               best={k:result["search"]["best"][k] for k in ("key","metrics")},
                               selected_graph=result["search"]["selected_graph"], optimizer=result["search"]["optimizer"]["backend"], artifacts=result["local"]["artifacts"])
        (out/"desktop-measurements.json").write_text(s.dump(records),encoding="utf8")
        dataset=ROOT/"examples/ai_round1/cmaq/emissions/A_EDGAR_CO_2014_Paris.nc"
        scan,_=Scanner().scan(dataset)
        cfg=(dataset.parent/"config_EDGAR.json").read_text(encoding="utf-8")
        state,local=s.prepare(scan,[(0,"primary")],"native",window.profiles["cmaq"],"gridded_emissions","NetCDF","","","",cfg)
        plan=approve(s.planning_summary(scan,[(0,"primary")],"native","NetCDF"),out,"conversion-plan")
        result=s.run(state,stage/"native-job",[])
        review=approve(s.review_summary(result),out,"conversion-review")
        (out/"conversion.zip").write_bytes(s.bundle(result,plan,review))
        records["conversion"]=dict(source=dataset.name, artifacts=result["local"]["artifacts"],
                                   native_ready=result["local"]["native_ready"])
    (out/"desktop-measurements.json").write_text(s.dump(records),encoding="utf8")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    acceptance(args.output,args.source)



