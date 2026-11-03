"""Bounded external sampling of the running experiment, never touches data."""
import argparse,json,time
from pathlib import Path
import psutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="output/beijing_mixed_gap_20261020/strict_grammar_v2/process_resources.jsonl")
    parser.add_argument("--entry",default="run_beijing_mixed_gap.py")
    args=parser.parse_args();out=Path(args.output)
    for _ in range(600):
        processes=[p for p in psutil.process_iter(["name","cmdline"]) if p.info["name"]=="python.exe" and
            len(p.info["cmdline"] or [])>=2 and Path(p.info["cmdline"][1]).name==args.entry]
        if not processes:break
        p=max(processes,key=lambda p:p.memory_info().rss);mem=p.memory_info();cpu=p.cpu_times()
        with out.open("a",encoding="utf-8") as f:f.write(json.dumps(dict(unix_time=time.time(),pid=p.pid,rss_bytes=mem.rss,
            process_lifetime_peak_bytes=mem.peak_wset,cpu_user=cpu.user,cpu_system=cpu.system))+"\n")
        time.sleep(2)


if __name__=="__main__":main()
