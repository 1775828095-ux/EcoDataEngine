"""Actual seed trajectories; no success-conditioned average or extrapolation."""
import sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from prepare_beijing_mixed_gap import OUT,save
sys.path.insert(0,"C:/Users/LX/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    save(OUT/"figure_contract.json",dict(question="How do search trajectories differ at actual cost, without treating structural scores as RMSE?",
        archetype="three separate quantitative single-panel views",source="best_so_far.csv; every seed, no smoothing or artificial trajectories",
        replicates="10 algorithm seeds per mechanism, not 10 independent environmental experiments",summary="raw trajectories; no aggregated uncertainty band",
        policy="undefined before first quality-passing graph; stop lines at actual attained budget; failed seeds retained in search_runs.csv",
        backend="Python/matplotlib (established workflow and saved preference)",export="183x105mm, PDF/SVG editable text, PNG preview, no journal acceptance claim"))
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":8,
        "pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
    data=pd.read_csv(OUT/"best_so_far.csv");colors={"A":"#666666","B":"#91689c","C":"#21759c","C-gap":"#c18539","R":"#399174"}
    for x,label in [("evaluation","Scored unique graph evaluations"),("real_executions","Unique real numerical executions"),("seconds","Elapsed search wall time (s)")]:
        fig,ax=plt.subplots(figsize=(7.2047244094,4.1338582677));fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.77)
        for mode,color in colors.items():
            for seed,g in data[data["mode"].eq(mode)].groupby("seed"):
                assert np.all(np.diff(g[x])>=0)
                valid=g[g.best.notna()]
                ax.step(valid[x],valid.best,where="post",color=color,alpha=.38,linewidth=.9)
        ax.set_xlabel(label);ax.set_ylabel("Best valid hierarchical RMSE (ug m-3)")
        handles=[Line2D([],[],color=c,label=m,linewidth=1.5) for m,c in colors.items()]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.98),ncol=5)
        fig.text(.12,.82,"Each line: one actual search seed; no line before a valid solution",fontsize=8)
        fig.canvas.draw()
        name=OUT/("search_curve_"+x)
        require_matplotlib_panel_alignment(fig,json_out=str(name)+".alignment.json",strict=True)
        fig.savefig(str(name)+".pdf");fig.savefig(str(name)+".svg");fig.savefig(str(name)+".png",dpi=300)
        plt.close(fig)


if __name__=="__main__":main()
