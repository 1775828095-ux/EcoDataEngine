# EcoData Engine 1.0 开发说明

## 第一次安装

系统要求：Windows 10/11 64位，推荐 Python 3.12.x 64位，建议至少 8 GB 内存和 5 GB 可用磁盘。安装 Python 时勾选 Python Launcher。联网后双击 `setup_dev.bat`，脚本会创建 `.venv`、安装锁定依赖并检查关键组件。

部分空间库使用 Rasterio/GeoPandas 随附的 GDAL、PROJ wheel，正常情况下无需单独安装 GDAL。可选 xESMF 原生后端需要 ESMF，Windows 默认不安装；项目会继续使用现有 fallback。QGIS、WASP GUI 不是启动本平台的必要条件。

## 启动与入口

双击 `run_dev.bat`。服务就绪后浏览器会打开 `http://127.0.0.1:8501`，启动失败查看 `logs/dev_startup.log`。程序入口是 `src/ecodata/interface/app.py`。

`EcoDataEngine_Windows_Portable.zip` 是给普通使用者的自包含版本：解压后直接双击 `启动 EcoData Engine.bat`，不需要另外安装 Python。源码开发仍使用 `setup_dev.bat`。

## 主要目录

- `src/ecodata/`：核心源码；`interface/` 是前端，`models/` 是模型 Schema、匹配与 Adapter。
- `tests/`：全部自动测试。
- `examples/`：测试夹具与精简真实案例。
- `docs/`：设计、能力边界与验证报告。
- `tools/`：交付包构建工具。

不要修改原始测试数据、`source_manifest.json`、质量基线或模型需求来让测试通过。不要提交 `.venv`、`logs`、`output`、`dist`、用户上传数据和密钥。

## 测试和打包

双击 `run_tests.bat` 执行完整 pytest。修改后先更新 `VERSION`，再双击 `build_windows.bat`。测试失败时构建立即停止；成功包位于 `dist/EcoDataEngine_Windows_Portable.zip`。

源码交付包由 `python tools/build_release.py source` 生成。真实回归必须至少覆盖 CSV/NetCDF/GeoTIFF/矢量转换、模型匹配、WASP Reader/数据包及工程用途匹配测试。

完整 Narragansett 原始归档约 2.3 GB，不在源码 ZIP 中重复打包；源码包保留回归实际使用的小型真实文件和清单。若研究完整工程，请从团队原始数据仓库恢复到 `examples/real_wasp_narragansett/`，不要覆盖精简回归样例。
