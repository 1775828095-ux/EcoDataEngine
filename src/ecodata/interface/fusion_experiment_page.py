"""Small user-facing experiment for observation-constrained reconstruction."""
from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile
from hashlib import sha256

import pandas as pd
import streamlit as st
import xarray as xr

from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, run_neural_reconstruction
from ecodata.fusion.neural_inputs import read_experiment_grid, read_experiment_stations
from ecodata.fusion.neural_experiment import run_reconstruction_comparison


ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "examples" / "datasets"


def _dataset(upload, **confirmations) -> xr.Dataset:
    return read_experiment_grid(upload.name, upload.getvalue(), **confirmations)


def _package(result) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("reconstructed_field.nc", result.field.to_netcdf())
        archive.writestr("validation_stations.csv", result.station_predictions.to_csv(index=False))
        archive.writestr("evaluation_report.json", json.dumps(result.report, ensure_ascii=False, indent=2, default=str))
    return buffer.getvalue()


def render_fusion_experiment_page() -> None:
    st.subheader("多源融合重构实验")
    st.caption("低分辨率背景场 + 辅助变量 + 训练监测站 → 高分辨率连续场。验证站不会参与训练。")
    source = st.radio("数据来源", ["内置PM2.5样例", "上传自己的数据"], horizontal=True)
    truth = None
    auxiliaries = []
    input_manifest = []
    if source == "内置PM2.5样例":
        background = xr.load_dataset(SAMPLE / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf")
        truth = xr.load_dataset(SAMPLE / "p3_pm25_truth_2p5km_48h.nc", engine="h5netcdf")
        stations = pd.concat([
            pd.read_csv(SAMPLE / "p3_pm25_train_stations.csv"),
            pd.read_csv(SAMPLE / "p3_pm25_validation_stations.csv"),
        ], ignore_index=True)
        st.info("内置样例已有固定划分：12个训练站、10个独立验证站。")
        input_manifest = [{'filename':name, 'sha256':sha256((SAMPLE/name).read_bytes()).hexdigest(), 'kind':'synthetic'}
                          for name in ('p3_pm25_coarse_10km_48h.nc', 'p3_pm25_truth_2p5km_48h.nc',
                                       'p3_pm25_train_stations.csv', 'p3_pm25_validation_stations.csv')]
    else:
        bg_file = st.file_uploader("低分辨率背景场 NetCDF / GeoTIFF", type=["nc", "nc4", "tif", "tiff"], key="neural_bg")
        station_file = st.file_uploader("监测站 CSV", type=["csv"], key="neural_station")
        truth_file = st.file_uploader("高分辨率参考真值 NetCDF（可选）", type=["nc", "nc4"], key="neural_truth")
        aux_files = st.file_uploader("辅助变量 NetCDF / GeoTIFF（可选，可多选）", type=["nc", "nc4", "tif", "tiff"],
                                     accept_multiple_files=True, key="neural_aux")
        if not bg_file or not station_file:
            st.info("监测站需要 station_id、time、latitude、longitude、value、unit；QA为qc_flag或quality_weight，可选observation_error为观测标准差。")
            return
        from ecodata.io.structures import read_structure, spatial_axes
        def confirmed_grid(upload, *, background=False):
            detected = read_structure(upload.name, upload.getvalue())
            x, y = spatial_axes(detected)
            names = [name for name, field in detected.data_vars.items() if {x,y}.issubset(field.dims) and field.dtype.kind in 'biuf']
            if not background and len(names)>1:
                names = st.multiselect(f'{upload.name}：辅助变量（可多选）', names, default=names, key=f'aux_vars_{upload.name}')
                if not names:
                    raise ValueError('请至少选择一个辅助变量')
                confirmed = []
                for name in names:
                    unit = None if detected[name].attrs.get('units') else st.text_input(f'{upload.name}/{name}：确认单位', key=f'unit_{upload.name}_{name}')
                    confirmed.append(_dataset(upload, variable=name, unit=unit))
                return xr.merge(confirmed)
            variable = st.selectbox(f'{upload.name}：选择变量', names, key=f'var_{upload.name}') if len(names)>1 else names[0] if names else None
            unit = None
            if variable and not detected[variable].attrs.get('units'):
                unit = st.text_input(f'{upload.name}：确认单位（不能为空）', key=f'unit_{upload.name}')
            timestamp = None
            if background and 'time' not in detected.dims:
                timestamp = st.text_input(f'{upload.name}：确认该背景场时刻', placeholder='文件真实时刻，如2021-02-01T00:00:00', key=f'time_{upload.name}') or None
                if timestamp is None:
                    raise ValueError('背景栅格没有时间，请确认真实时刻；不能从文件名猜测')
            categorical = st.checkbox(f'{upload.name}是分类土地利用（使用最近邻/one-hot）', key=f'cat_{upload.name}') if not background else False
            return _dataset(upload, variable=variable, unit=unit, timestamp=timestamp, categorical=categorical)
        try:
            background = confirmed_grid(bg_file, background=True)
            station_preview = pd.read_csv(io.BytesIO(station_file.getvalue()), dtype={'station_id': str})
            value_column = 'value'
            if value_column not in station_preview:
                excluded = {'station_id', 'time', 'latitude', 'longitude', 'quality_weight', 'observation_error'}
                options = [name for name in station_preview if name not in excluded and pd.api.types.is_numeric_dtype(station_preview[name])]
                value_column = st.selectbox('监测站观测值字段', options, key='neural_station_value')
            unit = None if 'unit' in station_preview else st.text_input('确认监测站观测单位', key='neural_station_unit')
            stations = read_experiment_stations(station_file.name, station_file.getvalue(), value_column=value_column, unit=unit)
            truth = confirmed_grid(truth_file) if truth_file else None
            auxiliaries = [confirmed_grid(item) for item in aux_files]
            input_manifest = [{'filename':item.name, 'sha256':sha256(item.getvalue()).hexdigest(), 'kind':'user_upload'}
                              for item in [bg_file, station_file, *([truth_file] if truth_file else []), *aux_files]]
        except Exception as exc:
            st.warning(f'请先确认输入信息：{exc}')
            return
    with st.expander("高级设置"):
        labels = {'fno': '原版FNO（基线）', 'multiscale_uno': '上一版UNO + Attention',
                  'observation_guided_uno': 'Observation-guided UNO（实验版本）'}
        backend = st.selectbox("重构版本", list(labels), index=0, format_func=labels.get)
        compare = st.checkbox('运行统一基线对照（双线性 / Kriging / QP / FNO / UNO）')
        validation_count = int(st.number_input("自动留出的验证站数量", min_value=1, value=20,
                                               help="如果CSV已有split列，则严格使用既有划分。"))
        epochs = int(st.slider("训练轮次", 10, 200, 40, 10))
        seed = int(st.number_input("随机种子", min_value=0, value=20261009))
        gate = st.slider('观测残差通路强度', 0., 1., .5, .1)
    if st.button("运行融合重构实验", type="primary"):
        st.session_state.pop('neural_reconstruction_result', None)
        st.session_state.pop('neural_comparison_result', None)
        try:
            with st.spinner("正在训练并用独立监测站验证……"):
                config = NeuralReconstructionConfig(epochs=epochs, backend=backend, observation_residual_gate=gate,
                                                     validation_station_count=validation_count, seed=seed)
                if compare:
                    experiment = run_reconstruction_comparison(background, stations, auxiliaries=auxiliaries,
                                                                evaluation_truth=truth, config=config)
                    experiment['report']['input_sources'] = input_manifest
                    st.session_state['neural_comparison_result'] = experiment
                else:
                    result = run_neural_reconstruction(
                        background, stations,
                        evaluation_truth=truth,
                        auxiliaries=auxiliaries,
                        config=config,
                    )
                    result.report['input_sources'] = input_manifest
                    st.session_state["neural_reconstruction_result"] = result
        except Exception as exc:
            st.error(f"实验未完成：{exc}")
            return
    experiment = st.session_state.get('neural_comparison_result')
    if experiment is not None:
        st.dataframe(experiment['comparison'], hide_index=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for method, field in experiment['fields'].items():
                archive.writestr(f'{method}.nc', field.to_netcdf())
            archive.writestr('comparison.csv', experiment['comparison'].to_csv(index=False))
            archive.writestr('validation_predictions.csv', experiment['predictions'].to_csv(index=False))
            archive.writestr('experiment_report.json', json.dumps(experiment['report'], ensure_ascii=False, indent=2, default=str))
        st.download_button('下载全部对照结果', buffer.getvalue(), 'neural_comparison.zip', 'application/zip')
        st.caption('PASS表示方法运行及输出检查通过，不代表精度提升；FAIL方法不生成结果文件。')
        return
    result = st.session_state.get("neural_reconstruction_result")
    if result is None:
        return
    metrics = result.report["metrics"]
    cols = st.columns(4)
    cols[0].metric("验证站", metrics["validation_station_count"])
    cols[1].metric("模型RMSE", f"{metrics['rmse']:.3f}")
    cols[2].metric("双线性RMSE", f"{metrics['bilinear_rmse']:.3f}")
    cols[3].metric("模型MAE", f"{metrics['mae']:.3f}")
    st.write({'Bias': metrics['bias'], '验证站峰值误差': metrics['validation_peak_error'],
              '空间相关性（无全场真值时不计算）': metrics['spatial_correlation']})
    if metrics["rmse"] < metrics["bilinear_rmse"]:
        st.success("本次独立站点验证优于双线性基线。")
    else:
        st.warning("本次没有优于双线性基线；不能宣称算法已优化。请检查数据量、训练轮次和变量关系。")
    if result.report["warning"]:
        st.warning(result.report["warning"])
    if metrics.get("full_field_rmse", 0) > metrics.get("bilinear_full_field_rmse", float("inf")):
        st.warning("独立参考场显示全场误差增加：站点改善不代表整个区域都改善。")
    st.line_chart(pd.DataFrame({"训练损失": result.loss_history}))
    st.dataframe(result.station_predictions.head(200), hide_index=True)
    st.download_button("下载实验结果ZIP", _package(result), "fusion_reconstruction_result.zip", "application/zip")
