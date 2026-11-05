"""Bounded C-shell heredoc analysis, never shell execution.

Reuse the project's static C-shell evaluator and f90nml, not a Bash interpreter.
Only literal heredocs to CAMx.in and a bounded arithmetic AWK idiom are admitted.
"""
import io,re
from pathlib import PurePosixPath
from .control_flow import ShellDeclarations,Value,logical_lines,parse
from .project_binding import CAMX_INPUTS,_values


class Templates(ShellDeclarations):
    def __init__(self,files):
        super().__init__(files);self.captures=[]

    def execute(self,nodes,env,control,conditional=False):
        if conditional and any(n[0]=='command' and '__ECODATA_CAPTURE__' in n[-1] for n in nodes):
            raise ValueError('CAMx模板激活条件未知，不能猜选分支')
        return super().execute(nodes,env,control,conditional)

    def expand(self,s,env):
        # C-shell filename root/extension modifiers apply to a scalar value.
        def modifier(m):
            v=env.get(m[1])
            if v is None or v.unknown:return '${UNRESOLVED_MODIFIER}'
            head,sep,tail=v.text.rpartition('.')
            return head if sep and m[2]=='r' else tail if sep else v.text
        return super().expand(re.sub(r'\$(\w+):([re])\b',modifier,s),env)

    def value(self,s,env):
        if s=='__ECODATA_CAPTURE__':
            self.captures.append(dict(env));return Value('captured')
        expanded=self.expand(s,env)
        if not expanded.unknown:
            match=re.fullmatch(r'''`echo\s+(\d+)\s*\|\s*awk\s+'\{printf\("%d",\$1([+-]\d+)\)\}'`''',expanded.text)
            if match:return Value(str(int(match[1])+int(match[2])))
        return super().value(s,env)


def inspect_template(files,control,text):
    import f90nml
    match=re.search(r'(?m)^\s*cat\s+<<\s*(\w+)\s*>\s*CAMx\.in\s*\r?\n',text)
    if not match:return None
    ending=re.search(r'(?m)^'+re.escape(match[1])+r'\s*$',text[match.end():])
    if ending is None:raise ValueError('CAMx heredoc未闭合')
    stop=match.end()+ending.start();tail=match.end()+ending.end()
    template=text[match.end():stop]
    if len(template)>1024*1024:raise ValueError('CAMx控制模板超过1MiB')
    engine=Templates(files)
    declaration=text[:match.start()]+'\nset ECODATA_CAPTURE = __ECODATA_CAPTURE__\n'+text[tail:]
    engine.execute(parse(logical_lines(declaration)),{},control)
    if not engine.captures:raise ValueError('CAMx没有确定执行路径的控制模板')
    runs=[];all_bindings=[];unknown=[];produced={}
    for env in engine.captures:
        # A shell heredoc expands even text quoted for the later Fortran reader.
        rendered=re.sub(r'\$\{\w+\}|\$\w+(?:\[[0-9]+\])?',
            lambda m:engine.expand(m[0],env).text,template)
        if re.search(r'\$|`',rendered):raise ValueError('CAMx模板包含未解析变量或命令，未生成控制文件')
        group=f90nml.read(io.StringIO(rendered))['camx_control']
        start=group.get('start_date_hour');restart=group.get('restart')
        if not isinstance(start,list) or len(start)!=4 or type(restart) is not bool:
            raise ValueError('CAMx日期/重启声明不明确')
        required={'chemistry_parameters','ozone_column','boundary_conditions'}
        if not required<=set(group) or not ({'photolyis_rates','photolysis_rates'}&set(group)):
            raise ValueError('CAMx核心输入声明缺失')
        restart_keys={'master_grid_restart','nested_grid_restart','pig_restart'}
        if restart and not group.get('master_grid_restart'):raise ValueError('重启缺少master_grid_restart')
        if not restart and not group.get('initial_conditions'):raise ValueError('冷启动缺少initial_conditions')
        keys=CAMX_INPUTS|restart_keys if restart else CAMX_INPUTS
        for key in sorted(keys&set(group)):
            if restart and key=='initial_conditions':continue
            for indexes,reference in _values(group[key]):
                if not reference:continue
                if not isinstance(reference,str):raise ValueError('CAMx文件引用不是字符串')
                row=dict(control=control,parameter=key,indexes=list(indexes),original=reference,day=start[:3],activation='ACTIVE')
                try:
                    target,reason=files.resolve(control,reference)
                    row.update(status='BOUND',target=target,sha256=files.identity(target)['sha256'],reason=reason)
                except ValueError as exc:
                    # Runtime restart output is a dependency, never pretend it exists.
                    expected=produced.get(reference) if key in restart_keys else None
                    row.update(status='GENERATED_DEPENDENCY' if expected else 'MISSING',reason=str(exc))
                    if expected:row['producer_required']=expected
                    else:unknown.append(str(exc))
                all_bindings.append(row)
        runs.append(dict(start=start,end=group.get('end_date_hour'),restart=restart,
            rendered_control=rendered,parameters=group.todict(),native_model_executed=False))
        output=group.get('root_output_name')
        if isinstance(output,str):
            for suffix in ('.inst','.finst'):
                produced[output+suffix]=dict(run_start=start,output_root=output,
                    native_model_executed=False,note='preceding declared output; requires successful integration')
    return dict(control=files.identity(control),bindings=all_bindings,unknown=unknown,
        runs=runs,parser='bounded C-shell declarations + f90nml '+f90nml.__version__,
        commands_executed=False,native_control_generated=True,source_modified=False,
        native_model_executed=False,trace=engine.trace,
        scope='Rendered derivative controls; runtime restart dependencies are not uploaded missing inputs')
