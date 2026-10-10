"""Import local examples via the real API. Existing projects are preserved."""
import json
import time
import sys
from pathlib import Path
import httpx

BASE='http://127.0.0.1:8018/api'
ROOT=Path(__file__).resolve().parents[1]

def main():
    with httpx.Client(base_url=BASE,timeout=60) as client:
        samples=client.get('/samples').json()
        existing=client.get('/projects').json()
        records=[]
        for sample in samples:
            # Repeat runs only skip imports by stable sample provenance stored in notes.
            marker='sample-id:'+sample['id']
            p=next((p for p in existing if marker in p['notes']),None)
            if p is None:
                r=client.post('/samples/'+sample['id']+'/import');r.raise_for_status();p=r.json()
                p=client.patch('/projects/'+p['id'],json={'notes':p['notes']+'\n'+marker}).json()
            status='needs-review'
            metadata={}
            if sample['kind']=='synthetic':
                if '--regenerate' in sys.argv or not p['artifact'] or p['stale']:
                    job=client.post('/projects/'+p['id']+'/jobs').json()
                    deadline=time.time()+90
                    while time.time()<deadline:
                        job=client.get('/jobs/'+job['id']).json()
                        if job['status'] not in {'queued','processing'}:break
                        time.sleep(.25)
                    status=job['status']
                    metadata=job.get('artifact',{}).get('metadata',{})
                else:
                    status='completed';metadata=p['artifact']['metadata']
            records.append({'sample':sample['id'],'kind':sample['kind'],'project_id':p['id'],'status':status,'elapsed_seconds':metadata.get('elapsed_seconds'),'asset_bytes':metadata.get('asset_bytes'),'worker_rss_mb':metadata.get('worker_rss_mb')})
            print(sample['id'],status,flush=True)
        out=ROOT/'artifacts';out.mkdir(exist_ok=True)
        (out/'seed-results.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
