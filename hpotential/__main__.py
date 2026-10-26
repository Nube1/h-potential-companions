"""Command line access to CRF scoring and population aggregation."""
import argparse, csv, json
from pathlib import Path
from .crf import H, ede


def main():
    parser=argparse.ArgumentParser(description='H-potential research scoring; not clinically calibrated')
    sub=parser.add_subparsers(dest='command',required=True)
    score=sub.add_parser('score',help='Score a CSV containing id,P,S,R columns')
    score.add_argument('input',type=Path);score.add_argument('output',type=Path)
    score.add_argument('--alpha',type=float,default=.5);score.add_argument('--rho',type=float,default=0)
    pop=sub.add_parser('ede',help='EDE of a scored CSV containing H; optional weight column')
    pop.add_argument('input',type=Path);pop.add_argument('--epsilon',type=float,default=0)
    args=parser.parse_args()
    with args.input.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if args.command=='score':
        if not rows: parser.error('Empty input')
        if args.input.resolve()==args.output.resolve(): parser.error('Output must differ from input')
        result=[]
        for row in rows:
            if set(row)!={'id','P','S','R'}:parser.error('Expected exactly id,P,S,R columns')
            h=H(*(float(row[k]) for k in ('P','S','R')),rho=args.rho,alpha=args.alpha)
            result.append({**row,'H':h,'alpha':args.alpha,'rho':args.rho})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(result)
        print(json.dumps({'rows':len(result),'output':str(args.output),'units':'same as supplied P'}))
    else:
        weights=[float(row['weight']) for row in rows] if rows and 'weight' in rows[0] else None
        print(json.dumps({'epsilon':args.epsilon,'ede':ede([float(row['H']) for row in rows],args.epsilon,weights),'n':len(rows)}))


if __name__=='__main__': main()
