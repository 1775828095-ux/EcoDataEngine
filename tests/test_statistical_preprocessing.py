"""Regression coverage for AI-proposed and locally verified preprocessing plans."""
def test_ai_plan_reaches_executable_preprocessing_and_export():
    import copy
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    import pandas as pd
    from ecodata.core.exceptions import EcoDataError

    from ecodata.desktop import discovery,recommendations,selected_processing,service,three_phase
    from ecodata.desktop.scan import Scanner
    from ecodata.desktop.statistical_preprocessing import PlanError,prepare


    with TemporaryDirectory(prefix="statistical-preprocessing-",dir=Path.cwd()) as name:
        root=Path(name)
        (root/"file.cio").write_text("SWAT+ fixture\nsimulation time.sim print.prt\nclimate null\n")
        (root/"time.sim").write_text("Title\nday_start yrc_start day_end yrc_end step\n1 2018 366 2020 0\n")
        (root/"print.prt").write_text("Title\nnyskip day_start yrc_start day_end yrc_end interval\n0 0 0 0 0 0\naa_int_cnt\n0\n")
        (root/"basin_res_aa.txt").write_text("Title\njday mon day yr unit gis_id name precip surface_area\nm^3 m^2\n1 1 1 2020 1 1 basin 60 1500 Original Simulation 0.000\n")
        (root/"basin_wb_yr.txt").write_text("Title\njday mon day yr unit gis_id name precip\nmm\n"+
            "".join(f"1 1 1 {year} 1 1 basin {value}\n" for year,value in [(2018,20),(2019,40),(2020,60)]))
        scan,_=Scanner().scan(root)
        catalogue,_=discovery.discover(scan,"swat_plus")
        def ref(filename,variable):
            index=next(i for i,row in enumerate(scan["rows"],1) if row["name"]==filename)
            field=next(j for j,f in enumerate(scan["rows"][index-1]["fields"],1) if f["name"]==variable)
            return f"D{index}.F{field}"
        volume=ref("basin_res_aa.txt","precip")
        depth=ref("basin_wb_yr.txt","precip")
        area=ref("basin_res_aa.txt","surface_area")
        plan=dict(start_year=2018,end_year=2020,target_period="average_annual",target_unit="mm",sources=[
            dict(field=volume,statistic="identity",area_field=area,evidence="同一输出对象的水量体积与表内面积"),
            dict(field=depth,statistic="mean",evidence="逐年累计量对共同年份取算术均值")])
        calls=[]
        def send(draft,*args,**kwargs):
            payload=json.loads(draft.summary)
            calls.append(draft.stage)
            if draft.stage=="file_classification":
                return dict(classifications=[dict(file=row["id"],kind="data" if row["field_count"] else "engineering")
                                             for row in payload["files"]])
            if "local_check_results" in payload:
                proof=next(item for item in payload["local_check_results"]["results"] if item["check"]=="preprocessing_plan")
                assert proof["status"]=="checked" and proof["output_records"]==[1,1],proof
                assert "60" not in json.dumps(proof),proof
            return dict(decisions=[dict(fields=[volume,depth],action="preprocess",physical_quantity="precipitation",
                goal="grid_comparison",**({} if "local_check_results" in payload else {"preprocessing_plan":plan}),algorithm_steps=[[
                    "temporal.window_align","units.area_normalize","units.harmonize"]],reason="统一年份后按对应面积换算")],
                questions=[dict(fields=[volume,depth],check="preprocessing_plan",question="完整三年窗口与面积能否核实？")])
        with patch.object(service,"send",side_effect=send):
            reply,checked,gated,candidates,approved,full=three_phase.run(scan,catalogue,{},False,"",lambda _:None,lambda _:None)
        assert calls==["file_classification","decision","decision"],calls
        assert ("C1","preprocess") in gated,checked
        request=recommendations.make_request(full,approved,checked,gated,include_samples=False)
        chosen=recommendations.from_decisions(request,reply["decisions"])
        assert chosen[0]["steps"]==["temporal.window_align","units.area_normalize","units.harmonize"],chosen
        task=dict(gated[("C1","preprocess")],recipes=[chosen[0]["steps"]])
        result=selected_processing.run(full,[("C1:preprocess",task)],root/"out",model_id="swat_plus")
        from io import BytesIO
        outputs=[pd.read_csv(BytesIO(raw)) for filename,raw in result["files"] if filename.endswith(".csv")]
        assert len(outputs)==2
        assert [frame.value.iloc[0] for frame in outputs]==[40,40]
        assert [frame.unit.iloc[0] for frame in outputs]==["mm","mm"]
        bad=copy.deepcopy(plan)
        bad["end_year"]=2021
        try:
            prepare(scan,bad,[volume,depth],catalogue["engineering_documents"])
        except PlanError:
            pass
        else:
            raise AssertionError("Unverified time window must be rejected")
        bad=copy.deepcopy(plan)
        bad["sources"][0]["area_field"]=depth
        try:
            prepare(scan,bad,[volume,depth],catalogue["engineering_documents"])
        except (PlanError,EcoDataError) as error:
            assert error
        else:
            raise AssertionError("Length must not be used as area")
        print("One-confirmation AI plan -> local evidence -> executable recommendation -> native preprocessing export passed",flush=True)

def test_leap_year_integration_and_evidenced_area_join():
    import copy
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    import pandas as pd

    from ecodata.desktop.scan import Scanner
    from ecodata.desktop.discovery import discover
    from ecodata.desktop.statistical_preprocessing import PlanError,prepare


    with TemporaryDirectory(prefix="window-preprocessing-",dir=Path.cwd()) as folder:
        root=Path(folder)
        times=pd.date_range("2019-01-01","2020-12-31")
        daily=pd.DataFrame([dict(date=str(date.date()),object_id=identity,rain=1.0)
                            for identity in ("001","002") for date in times])
        daily.columns=["date","object_id","rain [mm/day]"]
        daily.to_csv(root/"daily.csv",index=False)
        years=pd.DataFrame([dict(date=f"{year}-01-01",object_id=identity,depth=amount,volume=amount*(2 if identity=="002" else 1))
                           for identity in ("001","002") for year,amount in ((2019,365),(2020,366))])
        years.columns=["date","object_id","depth [mm]","volume [m^3]"]
        years.to_csv(root/"yearly.csv",index=False)
        (root/"areas.csv").write_text("object_id,surface [m^2]\n002,2000\n001,1000\n")
        (root/"README.md").write_text("# Model object inputs\n\nThe volume in yearly.csv uses the surface area in areas.csv, joined by object_id. Both share the same model object identifiers.\n")
        scan,_=Scanner().scan(root)
        catalogue,_=discover(scan,"swat_plus")
        def ref(filename,name):
            i=next(i for i,row in enumerate(scan["rows"],1) if row["name"]==filename)
            j=next(j for j,f in enumerate(scan["rows"][i-1]["fields"],1) if f["name"]==name)
            return f"D{i}.F{j}"
        amount=ref("yearly.csv","depth")
        rain=ref("daily.csv","rain")
        plan=dict(start_year=2019,end_year=2020,target_period="average_annual",target_unit="mm",sources=[
            dict(field=rain,statistic="sum",integrate_rate=True,input_period="daily",time_field=ref("daily.csv","date"),
                 object_field=ref("daily.csv","object_id"),evidence="日平均降水率按完整日历区间积分"),
            dict(field=amount,statistic="mean",input_period="yearly",time_field=ref("yearly.csv","date"),
                 object_field=ref("yearly.csv","object_id"),evidence="逐年累计量对两年取均值")])
        task=prepare(scan,plan,[rain,amount],catalogue["engineering_documents"])
        for frame in task["frames"]:
            np.testing.assert_allclose(frame.value,[365.5,365.5])
        volume=ref("yearly.csv","volume")
        plan["sources"][0]=dict(field=volume,statistic="mean",input_period="yearly",time_field=ref("yearly.csv","date"),
            object_field=ref("yearly.csv","object_id"),area_field=ref("areas.csv","surface"),
            area_object_field=ref("areas.csv","object_id"),evidence="按模型文档绑定的对象面积换算",evidence_documents=[
                doc["id"] for doc in catalogue["engineering_documents"] if doc.get("text") and "joined by" in doc["text"]])
        task=prepare(scan,plan,[volume,amount],catalogue["engineering_documents"])
        np.testing.assert_allclose(task["frames"][0].value,[365.5,365.5])
        bad=copy.deepcopy(plan)
        bad["sources"][0]["evidence_documents"]=[]
        try:
            prepare(scan,bad,[volume,amount],catalogue["engineering_documents"])
        except PlanError as error:
            assert "工程依据" in str(error)
        else:
            raise AssertionError("Cross-file area binding must have actual engineering evidence")
        years.iloc[1:].to_csv(root/"yearly.csv",index=False)
        incomplete,_=Scanner().scan(root)
        incomplete_catalogue,_=discover(incomplete,"swat_plus")
        try:
            prepare(incomplete,plan,[volume,amount],incomplete_catalogue["engineering_documents"])
        except PlanError as error:
            assert "完整覆盖" in str(error)
        else:
            raise AssertionError("A missing annual record must not silently change the average")
        print("Leap-year rate integration, two objects, explicit cross-file area joins and missing-evidence rejection passed",flush=True)
