"""Refresh the review inventory after an intentional source/artifact change."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    files={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(part in {'.git','.venv','.pytest_cache','__pycache__'} or part.endswith('.egg-info') for part in rel.parts):continue
        if str(rel) in ['manifest.json','verification.json']:continue
        files[str(rel)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT/'manifest.json').write_text(json.dumps({'status':'submission-prepared','date':'2026-10-13',
        'github_visibility':'public','github_url':'https://github.com/salehfarooq/cs5326-pa2-small-model-alignment',
        'files':files},indent=2)+'\n')
    print('Manifest:',len(files),'files')

if __name__=='__main__':main()
