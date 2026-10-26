import argparse
from pathlib import Path

from .config import load_config
from .data import synthetic_dataset
from .pipeline import run


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate-synthetic"); g.add_argument("--config",default="configs/study.yaml"); g.add_argument("--output",default="data/sample/synthetic.csv"); g.add_argument("--n",type=int,default=750)
    r=sub.add_parser("run"); r.add_argument("--config",default="configs/study.yaml"); r.add_argument("--data",required=True); r.add_argument("--output",default="outputs")
    a=p.parse_args(); cfg=load_config(a.config)
    if a.command=="generate-synthetic":
        dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True); synthetic_dataset(cfg,a.n,cfg.seed).to_csv(dest,index=False); print(dest)
    else: print(run(a.data,cfg,a.output).to_string(index=False))

if __name__=="__main__": main()
