"""Single local CPU worker. A filesystem lock prevents competing worker lifespans."""
import fcntl
import json
import logging
import signal
import resource
import sys
import time
import traceback
import psutil
from . import store

log=logging.getLogger('eyewear.worker')

def process_next():
    with store.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row:return False
        job=json.loads(row['payload']);snapshot=json.loads(row['snapshot'])
        job.update(status='processing',stage='Extracting and fitting photo-derived geometry')
        conn.execute('UPDATE jobs SET status=?,payload=? WHERE id=?',(job['status'],json.dumps(job),job['id']))
    from .reconstruction import reconstruct,NeedsCorrection
    start=time.perf_counter()
    cpu_start=time.process_time()
    try:
        folder=store.DATA/'artifacts'/job['id'];folder.mkdir(parents=True,exist_ok=True)
        metadata=reconstruct(snapshot,folder)
        metadata['elapsed_seconds']=round(time.perf_counter()-start,3)
        metadata['cpu_seconds']=round(time.process_time()-cpu_start,3)
        metadata['worker_rss_mb']=round(psutil.Process().memory_info().rss/1024**2,1)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata['worker_lifetime_peak_rss_mb']=round(peak/(1024**2 if sys.platform=='darwin' else 1024),1)
        metadata['input_revision']=snapshot['revision']
        metadata['asset_bytes']=(folder/'model.glb').stat().st_size
        metadata['validation_status']='Not physically validated; review source and withheld views.'
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        artifact=dict(job_id=job['id'],revision=snapshot['revision'],model_url=f"/api/artifacts/{job['id']}/model.glb",metadata_url=f"/api/artifacts/{job['id']}/metadata.json",metadata=metadata)
        store.mutate_project(job['project_id'],lambda p:p.update(artifact=artifact))
        job.update(status='completed',stage='Ready for review',artifact=artifact)
    except NeedsCorrection as e:
        job.update(status='needs-correction',stage='Merchant correction required',error=str(e),details=getattr(e,'details',None))
    except Exception as e:
        log.error('Job %s failed: %s\n%s',job['id'],e,traceback.format_exc())
        job.update(status='failed',stage='Reconstruction failed',error=f'{type(e).__name__}: {e}')
    job['finished_at']=store.now()
    store.save_job(job)
    return True

def main():
    logging.basicConfig(level=logging.INFO)
    store.initialize()
    with (store.DATA/'worker.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            log.info('Another worker owns the local queue.');return
        with store.db() as conn:
            for row in conn.execute("SELECT payload FROM jobs WHERE status='processing'").fetchall():
                job=json.loads(row['payload']);job.update(status='failed',stage='Interrupted by restart',error='Worker stopped before finishing. Generate again to retry.')
                conn.execute('UPDATE jobs SET status=?,payload=? WHERE id=?',(job['status'],json.dumps(job),job['id']))
        running=True
        def stop(*_):
            nonlocal running
            running=False
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while running:
            if not process_next():time.sleep(.4)

if __name__=='__main__':main()
