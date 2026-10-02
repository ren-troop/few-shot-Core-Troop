"""Check the staged index before committing. Does not modify files or Git history."""
import argparse
import json
import re
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pt','.pth','.ckpt','.png','.pdf','.zip','.mat'}


def issues_for(path, content):
    p=Path(path);parts=p.parts;issues=[]
    if any(x.startswith('outputs') or x in ('__pycache__','.venv','venv','.idea') for x in parts):
        issues.append('runtime directory')
    if p.suffix.lower() in FORBIDDEN:issues.append('binary / large artifact')
    if '.csv' in p.name.lower() and not (parts and parts[0]=='results' and p.suffix.lower()=='.csv'):
        issues.append('CSV outside results/')
    allowed={'water_label_missing_ready/data/manifest.json',
             'water_label_missing_ready/data/water_quality_uci/metadata.json',
             'water_label_missing_ready/data/water_potability/metadata.json'}
    if 'data' in parts and p.as_posix() not in allowed:issues.append('raw/generated data')
    if len(content)>2_000_000:issues.append('file exceeds 2 MB')
    if p.suffix=='.json':
        try:s=content.decode('utf-8');json.loads(s)
        except (ValueError,UnicodeError):issues.append('invalid JSON');s=''
        if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/|/workspace/|/tmp/',s):issues.append('machine-specific absolute path in JSON')
    return issues


def audit(index=False):
    bad=[];count=0
    if index:
        names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMRT','-z'],cwd=ROOT).decode('utf-8').split('\0')
        entries=((n,subprocess.check_output(['git','show',':'+n],cwd=ROOT)) for n in names if n)
    else:
        # File-tree mode is intended for the clean extracted delivery, before running experiments.
        entries=((f.relative_to(ROOT).as_posix(),f.read_bytes()) for f in ROOT.rglob('*')
                 if f.is_file() and '.git' not in f.relative_to(ROOT).parts and '__pycache__' not in f.parts)
    for name,data in entries:
        count+=1
        for reason in issues_for(name,data):bad.append((name,reason))
    for name,reason in bad:print(f'FAIL {name}: {reason}')
    print(f'Checked {count} files; issues={len(bad)}; source={"staged changes" if index else "file tree"}')
    return bool(bad)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--git-index',action='store_true')
    raise SystemExit(audit(p.parse_args().git_index))
