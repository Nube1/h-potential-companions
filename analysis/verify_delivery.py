"""Check deliverable integrity and claims against generated numerical outputs."""
import csv, json, hashlib
from pathlib import Path
from pypdf import PdfReader

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'results/manifest.json').read_text())
for name,digest in manifest['csv_sha256'].items():
    assert hashlib.sha256((root/'results'/name).read_bytes()).hexdigest()==digest,name
with (root/'results/measurement.csv').open() as f:rows=list(csv.DictReader(f))
drift=next(x for x in rows if x['scenario']=='social_intercept_shift' and x['domain']=='2')
assert abs(float(drift['coverage95'])-.72)<1e-12
assert abs(float(drift['bias'])-.5006630333845059)<1e-10
for paper in sorted((root/'papers').glob('*.pdf')):
    reader=PdfReader(paper)
    text='\n'.join(p.extract_text() for p in reader.pages)
    assert 'Abstract' in text and 'References' in text
    assert len(reader.pages)>=4
    assert paper.with_suffix('.md').exists() and paper.with_suffix('.tex').exists()
    print(paper.name,len(reader.pages),'pages')
print('CSV hashes, key numeric claims and PDF completeness verified')
