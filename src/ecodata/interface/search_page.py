"""Existing Streamlit interface, thin client of PlatformSearch/v2."""
from pathlib import Path
import json,uuid,hashlib
import streamlit as st
from ecodata.algorithms.platform_search import SearchInput,submit_search,process_current_input,EngineeringMetric,DatasetReference,parse_inputs,CURRENT_INPUT_BUDGET
from ecodata.interface.temporal_search_page import download_search_result


def process_search_upload(inputs,request,evaluation,folder,**options):
    if request.get('decision_policy')=='current_input':
        if evaluation is not None or options.pop('validation',None) is not None:raise ValueError('Current-input page does not pass external selection labels')
        if options.pop('mode','PortraitMutation')!='PortraitMutation':raise ValueError('Current-input page uses the random mutation service')
        return process_current_input(inputs,request,folder,**options)
    return submit_search(inputs,request,evaluation,folder,**options)


def request_fingerprint(inputs,goal):
    from ecodata.algorithms import platform_search
    from ecodata.algorithms.search_stopping import policy
    from ecodata.algorithms.reliable_selection import POLICY
    from ecodata.algorithms.development_validation import DEFAULT_REGISTRY
    publications={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'UNPUBLISHED'
                  for p in (POLICY,DEFAULT_REGISTRY)}
    h=hashlib.sha256(json.dumps(dict(goal,platform_version=platform_search.VERSION,stopping_policy=policy({}).get('sha256','UNPUBLISHED'),selection_publications=publications),sort_keys=True,default=str).encode())
    for item in inputs:
        h.update(json.dumps([item.filename,item.role,item.metadata],sort_keys=True,default=str).encode());h.update(item.content)
    return h.hexdigest()


def render_search_page():
    st.subheader("统一算法选择与组合 · 空间 / 时间 / 多源")
    st.caption("画像短组合枚举与历史GOLEM模式共用真实执行和质量门。通用NetCDF不等于模拟器原生输入。")
    selection_mode=st.selectbox("选择机制",["PortraitMutation","Autonomous","AutonomousField","Profile","C-local"],format_func=lambda v:{"PortraitMutation":"当前数据自主判断 · 随机变异搜索","Autonomous":"自主组合 · 全注册库与统一真实误差","AutonomousField":"互补时空场 · 开放图与可见锚点验证","Profile":"模型需求与三画像 · 短组合 / 可选遗传寻优","C-local":"历史GOLEM局部搜索"}[v],key="unified_selection_mode")
    model_profile=None;selected_field=None
    if selection_mode=="Autonomous":
        st.caption("自主模式扫描全注册库，按实际接口逐层组合，不使用手写流程或画像软分。当前自动效果评价只支持单站逐日重建：从可见值构造隔离搜索与确认窗口；无可用参考时明确停止。资源用尽只报告当前最好，不能当作全局最优。")
    if selection_mode=="AutonomousField":
        st.caption("上传稀疏精细场及连续粗场；至少三张精细锚点时平台自动隐藏中间锚点选图，再用全部可见锚点执行。目标JSON须声明时间、网格及有依据的complementary_observation_operator。无可见参考不虚构准确率；资源停止不代表全局最优。")
    if selection_mode=="PortraitMutation":
        st.caption("三画像软评分默认关闭；保留物理硬约束、随机变异和真实验证选优。")
        st.caption("按当前输入随机构图，三画像给出有依据的候选优先级，保留随机探索。验证匹配缺口的边界、时间与可见邻域；支撑不足明确提示。每条候选真实执行，最终按验证效果选图。")
        st.caption("多站点、站点与空间场、空间位置集合及协变量可走空间估计入口，按留位置验证选组合；模型硬约束继续核验。新空间端口的软收益尚未校准，不套用已有任务的权重。")
        st.caption("先保存合格产品，再随机变异；按开发校准的收益停滞规则停止。搜索与部署分别设保护额度，均记入总成本，超时自动收尾。质量门固定执行，资源停止不代表全局最优。")
    if selection_mode in {"Profile","Autonomous","PortraitMutation"}:
        from ecodata.algorithms.profile_matching import MODELS,model_table
        model_id=st.selectbox("下游模型",MODELS,key="portrait_model")
        table=model_table(model_id)
        from ecodata.models.requirements import GENERIC_UNITS
        available=[dict(r,fields=[f for f in r['fields'] if f.get('data_type',f.get('kind','continuous')) in {'continuous','categorical'}
                    and f.get('unit','').casefold() not in GENERIC_UNITS]) for r in table['roles'] if not r['record_only']]
        available=[r for r in available if r['fields']]
        if available:
            role_name=st.selectbox("本次准备的输入角色",[r['role'] for r in available],key="portrait_role")
            role=next(r for r in available if r['role']==role_name)
            field=st.selectbox("已记录的字段契约",[f['name'] for f in role['fields']],key="portrait_field")
            selected_field=next(f for f in role['fields'] if f['name']==field)
            model_profile=dict(id=model_id,role=role_name,field=field)
        from ecodata.algorithms.platform_search import model_portrait_template
        with st.expander("模型画像：已核实需求与待确认偏好"):
            draft=model_portrait_template(model_id)
            st.json(draft)
            st.download_button("下载模型画像证据模板",json.dumps(draft,ensure_ascii=False,indent=2),
                model_id+"_portrait_draft.json","application/json",key="model_portrait_template_download")
            st.caption("文档和源码字段不等于科学敏感性。场景阈值、权重及变量关系须有依据；通过公共注册接口发布后，在补充需求JSON的portrait_policy选择版本与场景。")
        st.caption("字段来自现有准备契约；未知敏感性不打分。一次字段任务不等于完整模型输入或原生运行验收。")
        with st.expander("模型需求表及证据范围"):st.json(table)
    task=st.selectbox("处理任务",["spatial","temporal","multisource"],index=2 if selection_mode in {"AutonomousField","PortraitMutation"} else (1 if selection_mode=="Autonomous" else 0),format_func=lambda t:{"spatial":"空间处理","temporal":"时间聚合 / 插值","multisource":"多源融合 / 协变量估计"}[t],key="unified_task")
    uploads=st.file_uploader("一个或多个已支持数据文件",type=["nc","nc4","csv","tsv","tif","xlsx"],accept_multiple_files=True,key="unified_inputs")
    inputs=[]
    for index,upload in enumerate(uploads or []):
        role=st.selectbox("用途："+upload.name,["primary","background","auxiliary","observations","query","covariates","state","target","training_reference","historical_simulation"],key=f"unified_role_{index}")
        declaration=st.text_area("该文件缺少元数据时的显式声明（JSON，可留{}；不覆盖冲突原值）",value="{}",key=f"unified_meta_{index}")
        try:inputs.append(SearchInput(upload.name,role,upload.getvalue(),json.loads(declaration)))
        except Exception as exc:st.error(str(exc))
    variable=st.text_input("目标变量",value=selected_field['name'] if selected_field else "aod",key="unified_variable")
    unit=st.text_input("已确认单位（未知填UNKNOWN）",value=selected_field['unit'] if selected_field else "UNKNOWN",key="unified_unit")
    semantic=st.selectbox("变量性质",["continuous","categorical","accumulation","cell_total","multidimensional_array"],key="unified_semantic")
    purpose=st.selectbox("多源含义",["same_variable","station_covariates","features"],key="unified_purpose") if task=="multisource" else None
    nonnegative=st.checkbox("物理上非负",value=False,key="unified_nonnegative")
    allow_training=st.checkbox("允许使用本次明确提供的训练资源拟合（不包含最终测试）",value=False,key="unified_training")
    allow_weights=st.checkbox("允许使用平台已有、且通过任务契约核验的模型权重",value=False,key="unified_weights")
    st.caption("目标坐标、时间频率、区间语义及必要资源按科学定义填写。Profile模式可设置model_requirements、preferences及profile_options；候选算法池仅在profile_options.candidate_pool.algorithms显式冻结，不指定预期赢家或图模板。")
    st.caption("多源编排可在补充JSON配置composition：source_stages指定逐源阶段候选与先后约束，windows指定时段/重叠/上下文，fusion_options指定主辅角色与权重候选，post_stages指定融合后步骤，coordination指定有依据的协调门。先三画像筛选再独立变异，不依赖GOLEM；组合数随本次配置精确计算。")
    st.caption("互补时空数据可在composition.parallel_scoring声明weights（data、model、collaboration）及evidence，启用数据适用性、模型偏好、组合协同三路评分。平台遮挡输入中可用的内部细网格观测并真实执行局部组合；缺少夹持观测时保留未知。探测执行成本另列，模型输入偏好不等于模拟器精度。")
    st.caption("共同强基准模式在parallel_scoring.common_reference声明开发冻结文件的path和sha256；collaboration权重须为0，收益仅作判断及主评分同分时比较，不再抵消较差主评分。不提供该声明时保留旧加权路径用于兼容与对照。")
    if selection_mode=="Profile":
        st.caption("三表联合评分决定推荐。内置五项有依据的性质，可通过scoring.registry_version选择属性版本、scoring.calibration_id加载已校准参数。关闭软加分不会取消硬要求；未知属性保留区间。")
        st.caption("无独立参考时不验收精度。启用站点探测评分会执行训练站分折，成本单列。旧学习预测仅供诊断；没有声明评分偏好时按成本安排。实际误差最优另列为事后对照，不冒充推荐。")
    target=st.text_area("目标及补充需求JSON",value='{"target_lat": null, "target_lon": null, "parameters": {}}',key="unified_target")
    st.caption("可选feature_declarations须含source_id、feature、value、unit、status、confirmed及evidence，不能覆盖文件事实。optional_scoring仅接受平台模型ID；受限开发模型须显式acknowledge_limited_development，未知收益不等于零收益。")
    reference=st.file_uploader("可选validation参考NetCDF（与处理输入分开，仅用于独立评价）",type=["nc","nc4"],key="unified_reference")
    search_labels=st.file_uploader("可选搜索验证标签NetCDF（仅组合模式，用于变异反馈；变量须标split=search_validation，禁止最终测试）",type=["nc","nc4"],key="unified_search_validation")
    role_confirm=st.checkbox("参考是合法验证数据，不是最终测试；拟合数据与参考标签隔离",key="unified_reference_role")
    metric=st.selectbox("无参考时的依据",["none","coverage"],format_func=lambda v:{"none":"按画像适配推荐，精度未评价","coverage":"实际有效覆盖，不代表精度"}[v],key="unified_metric")
    quality_confirm=st.checkbox("确认保留原始缺失，并查看质量警告和UNKNOWN",key="unified_quality_confirm")
    try:composition_enabled="composition" in json.loads(target)
    except (ValueError,TypeError):composition_enabled=False
    open_search=composition_enabled or selection_mode in {"Autonomous","AutonomousField","PortraitMutation"}
    evaluations=int(st.number_input("总算法调用上限（含验证、搜索和最终执行）" if selection_mode=="PortraitMutation" else "候选执行资源预算",1,10000 if open_search else 20,CURRENT_INPUT_BUDGET['objective_limit'] if selection_mode=="PortraitMutation" else 5,key="unified_evaluations"))
    proposals=int(st.number_input("图提议资源预算",1,1000000 if open_search else 100,CURRENT_INPUT_BUDGET['proposal_limit'] if selection_mode=="PortraitMutation" else 50,key="unified_proposals"))
    seconds=int(st.number_input("搜索墙钟上限（秒，正在执行的内核返回后停止）",1,3600 if open_search else 60,CURRENT_INPUT_BUDGET['wall_seconds'] if selection_mode=="PortraitMutation" else 30,key="unified_seconds"))
    seed=int(st.number_input('随机种子（相同输入和设置可复现）',0,2147483647,11,key='unified_seed'))
    if st.button("识别输入",key="unified_inspect"):
        try:
            _,audit=parse_inputs(inputs);st.session_state["unified_identification"]=audit
        except Exception as exc:st.error(str(exc))
    if "unified_identification" in st.session_state:
        with st.expander("输入识别与待确认语义"):st.json(st.session_state["unified_identification"])
    current_identity=request_fingerprint(inputs,dict(task=task,variable=variable,unit=unit,semantic=semantic,target=target,
        mode=selection_mode,model_profile=model_profile,nonnegative=nonnegative,seed=seed,
        budget=[evaluations,proposals,seconds],allow_training=allow_training,allow_weights=allow_weights))
    if st.button("提交统一搜索",key="unified_submit"):
        folder=Path(__file__).resolve().parents[3]/"output/platform_search_requests"/uuid.uuid4().hex
        try:
            if not quality_confirm:raise ValueError("请先确认只读质量检查及保留缺失")
            request=dict(json.loads(target),task=task,variable=variable,unit=unit,semantic_type=semantic,nonnegative=nonnegative,allow_training=allow_training,allow_existing_weights=allow_weights)
            if selection_mode in {"Profile","Autonomous","PortraitMutation"} and 'application_contract' not in request:
                if model_profile is None:raise ValueError("当前模型没有可执行的已记录字段契约")
                request['model_profile']=model_profile
            if purpose:request["purpose"]=purpose
            if purpose=="station_covariates":request["adapter"]="station_covariates"
            if reference:
                if not role_confirm:raise ValueError("需要确认验证参考用途及隔离")
                from ecodata.recognition import recognise_file
                ds,_=recognise_file(reference.name,reference.getvalue())
                evaluation=DatasetReference(ds[variable],variable)
            else:evaluation=EngineeringMetric(variable) if metric=="coverage" else None
            progress=st.empty()
            validation=None
            if search_labels is not None:
                from ecodata.algorithms.platform_search import SearchValidation
                from ecodata.recognition import recognise_file
                label_data,_=recognise_file(search_labels.name,search_labels.getvalue())
                validation=SearchValidation(label_data[variable],variable)
            elif selection_mode=='PortraitMutation':
                request.update(decision_policy='current_input',mutation_control='random')
                evaluation=None
            result=process_search_upload(inputs,request,evaluation,folder,mode=selection_mode,seed=seed,budget=dict(objective_limit=evaluations,total_call_limit=evaluations if selection_mode=='PortraitMutation' else None,probe_call_limit=min(CURRENT_INPUT_BUDGET['probe_call_limit'],evaluations//3) if selection_mode=='PortraitMutation' else CURRENT_INPUT_BUDGET['probe_call_limit'],proposal_limit=proposals,callback_limit=100,wall_seconds=seconds),progress=lambda event:progress.json(event),validation=validation)
            st.session_state["unified_result"]=(result,str(folder))
        except Exception as exc:st.session_state["unified_result"]=({"status":"BLOCKED","error":str(exc)},str(folder))
        st.session_state['unified_result_identity']=current_identity
    if 'unified_result' in st.session_state and st.session_state.get('unified_result_identity')!=current_identity:
        st.info('输入或目标已改变，请重新提交；上一批结果不作为当前输入的处理结果。')
    if "unified_result" in st.session_state and st.session_state.get('unified_result_identity')==current_identity:
        result,folder=st.session_state["unified_result"]
        if result.get("error"):st.error(result["error"])
        elif result.get("stop_reason")=="RECOMMENDATION_ONLY":st.info("仅生成契约相容的候选推荐；未执行、未评价精度。")
        elif result.get('status')=='VALIDATION_EVIDENCE_INSUFFICIENT':st.warning(result.get('reason','任务验证证据不足'))
        elif result.get("best"):st.success("已生成通过格式与物理质量检查的结果；精度证据见任务验证方式")
        else:st.warning("预算内没有合格处理图；查看停止及失败记录")
        st.write("停止原因",result.get("stop_reason","未执行"))
        st.write("评价口径",result.get("evaluation_protocol",{}))
        if result.get('self_validation_protocol') or result.get('task_acceptance'):
            with st.expander("任务验证方式与证据范围",expanded=True):
                protocol=result.get('self_validation_protocol') or {}
                st.json(dict(tasks=protocol.get('tasks'),support_level=protocol.get('support_level'),
                    task_support=protocol.get('task_support'),acceptance=result.get('task_acceptance')))
        if result.get('candidate_prior'):
            with st.expander('候选画像优先级与最终执行保障'):
                st.json(dict(candidate_prior=result['candidate_prior'],final_validation_best_delivered=result.get('final_validation_best_delivered'),
                    reserved_calls=result.get('total_cost',{}).get('deployment_call_reserve'),reserved_seconds=result.get('total_cost',{}).get('deployment_time_reserve_seconds')))
        if selection_mode=="PortraitMutation":
            with st.expander("响应评分、科学去重和总成本"):
                cost=result.get('total_cost',{})
                st.json(dict(parent_policy=result.get('parent_policy'),response_score_intervals=result.get('response_score_intervals'),
                    scientific_state_deduplications=result.get('scientific_state_deduplications'),
                    call_limit=cost.get('limit'),new_calls=cost.get('used'),phases=cost.get('phases'),
                    fit_adjust_calls=cost.get('fit_adjust_calls'),failed_calls=cost.get('failed_calls'),service_seconds=cost.get('service_seconds')))
        with st.expander("数据问题、必需/可选能力与定向操作"):
            st.json(dict(problems=result.get("problems",[]),directed_config=result.get("directed_config",{}),
                optional_benefit_model=result.get('optional_benefit_model'),optional_benefit_inference_seconds=result.get('optional_benefit_inference_seconds'),
                note='可选收益只调整探索概率；最终指标来自真实执行。开发模型不代表可靠泛化。'))
        best=result.get("best") or {}
        if result.get("version")=="source_time_composition_v1":
            with st.expander("逐源角色、先后顺序、时段安排与协调验收"):
                st.json(dict(plan=best.get("plan"),coordination=best.get("coordination"),
                    initial_score=result.get("initial_best_score"),final_score=result.get("final_best_score"),
                    mutation_score=result.get("mutation_best_score"),mutation_improvement=result.get("mutation_improvement"),
                    mutations=result.get("proposals"),certification=result.get("certification")))
        if selection_mode=="Profile":
            if result.get("recommendation_certainty")=="UNRESOLVED_REQUIRES_PROBE":
                st.warning("候选适配范围重叠或存在未知，尚不能确认最优；当前推荐不等于精度保证。")
            if result.get("state_evolution"):
                with st.expander("多维遗传变异、Pareto候选及限定域核验"):
                    st.json(result["state_evolution"])
            with st.expander("三表评分依据、模型需求与组合收益"):
                st.json(dict(matching_status=result.get('matching_status'),
                    candidate_pool=result.get('candidate_pool'),
                    model_requirements=result.get('portraits',{}).get('model',{}).get('matching_requirements'),
                    matching=best.get('matching'),predicted_metrics=best.get('predicted_metrics'),
                    combination_gains=result.get('combination_gains',[])))
        st.json(dict(counts=result.get("counts",{}),discrimination=result.get("discrimination"),
            best={key:best[key] for key in ("objective","RMSE","MAE","Bias","accuracy_assessed","file_readback","product") if key in best}))
        if best.get("graph"):
            with st.expander("选中处理图及实际端口连接"):st.json(best["graph"])
        elif result.get('selected_graph'):
            with st.expander('平台选中的空间处理图及端口连接'):st.json(result['selected_graph'])
        with st.expander("完整搜索轨迹、失败及成本"):st.json(result)
        for filename in ("request.json","decision.json"):
            path=Path(folder)/filename
            if path.is_file():
                with st.expander(filename):st.json(json.loads(path.read_text(encoding="utf-8")))
        st.download_button("下载实际产品、图与决策记录",download_search_result(folder),"platform_search.zip","application/zip",key="unified_download")
