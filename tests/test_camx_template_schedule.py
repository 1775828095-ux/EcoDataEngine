import io,zipfile,contextlib
import pytest
from ecodata.models.project_binding import ProjectFiles
from ecodata.models.camx_templates import inspect_template


SCRIPT='''set MET = "../met"
set RESTART = "NO"
foreach today (10.162 11.163)
set CAL = $today:r
set JUL = $today:e
set CALDAY = 201606${CAL}
set YESTERDAY = `echo ${CALDAY} | awk '{printf("%d",$1-1)}'`
if ( $RESTART == "NO" ) then
set RESTART = "false"
else
set RESTART = "true"
endif
cat << EOF > CAMx.in
&CAMx_Control
start_date_hour=2016,6,${CAL},0,
end_date_hour=2016,6,${CAL},2400,
restart=.${RESTART}.,
chemistry_parameters='$MET/chem',
photolysis_rates='$MET/photo',
ozone_column='$MET/ozone',
boundary_conditions='$MET/boundary.${CALDAY}',
initial_conditions='$MET/ic',
root_output_name='../out/restart.${CALDAY}',
master_grid_restart='../out/restart.${YESTERDAY}.inst',
/
EOF
set RESTART = "YES"
end
'''


def analyze(script):
    data=io.BytesIO();data.name='case.zip'
    with zipfile.ZipFile(data,'w') as z:
        z.writestr('project/runfiles/run.job',script)
        for name in ['chem','photo','ozone','ic','boundary.20160610','boundary.20160611']:
            z.writestr('project/met/'+name,b'original')
    with contextlib.ExitStack() as stack:
        files=ProjectFiles([data],stack)
        return inspect_template(files,'project/runfiles/run.job',script)


def test_two_days_and_restart_are_not_confused_with_missing_uploads():
    r=analyze(SCRIPT)
    assert [d['restart'] for d in r['runs']]==[False,True]
    assert [d['start'] for d in r['runs']]==[[2016,6,10,0],[2016,6,11,0]]
    assert not r['unknown'] and not r['commands_executed']
    dependencies=[b for b in r['bindings'] if b['status']=='GENERATED_DEPENDENCY']
    assert len(dependencies)==1 and dependencies[0]['original']=='../out/restart.20160610.inst'


def test_arbitrary_command_substitution_is_rejected():
    with pytest.raises(ValueError,match='未解析'):
        analyze(SCRIPT.replace('set CAL = $today:r','set CAL = `curl example.com`'))


def test_unknown_branch_does_not_render_an_assumed_control():
    with pytest.raises(ValueError,match='激活条件未知'):
        analyze('if ( $UNDEFINED == yes ) then\n'+SCRIPT+'\nendif')


def test_missing_input_is_reported_without_substituting_another_name():
    r=analyze(SCRIPT.replace("'$MET/chem'","'$MET/missing'"))
    assert any(b['status']=='MISSING' for b in r['bindings']) and r['unknown']
