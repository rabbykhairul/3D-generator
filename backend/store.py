"""Small durable local store; each operation uses its own SQLite connection."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .engine_version import ENGINE_VERSION

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EYEWEAR_DATA_DIR', ROOT / 'data')).resolve()

def now():
    return datetime.now(timezone.utc).isoformat()

def uid():
    return uuid4().hex

@contextmanager
def db():
    DATA.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATA / 'studio.sqlite3', timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def initialize():
    with db() as conn:
        conn.executescript('''
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
            status TEXT NOT NULL, payload TEXT NOT NULL, snapshot TEXT NOT NULL, created_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS detection_runs(id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
            photo_id TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS detection_runs_project ON detection_runs(project_id, created_at);
        ''')

def get_project(project_id):
    with db() as conn:
        row = conn.execute('SELECT payload FROM projects WHERE id=?', (project_id,)).fetchone()
    return json.loads(row['payload']) if row else None

def record_detection(project_id, photo_id, report):
    """Retain bounded diagnostics, not image copies, labels or project mutations."""
    report = dict(report, id=uid(), created_at=now())
    with db() as conn:
        conn.execute('INSERT INTO detection_runs VALUES (?,?,?,?,?)',
                     (report['id'], project_id, photo_id, report['created_at'], json.dumps(report)))
        conn.execute('DELETE FROM detection_runs WHERE project_id=? AND id NOT IN '
                     '(SELECT id FROM detection_runs WHERE project_id=? ORDER BY created_at DESC LIMIT 100)',
                     (project_id, project_id))
    return report

def detection_history(project_id, photo_ids):
    with db() as conn:
        rows = conn.execute('SELECT photo_id,payload FROM detection_runs WHERE project_id=? '
                            'ORDER BY created_at DESC LIMIT 100', (project_id,)).fetchall()
    return [json.loads(r['payload']) for r in rows if r['photo_id'] in photo_ids]

def list_projects():
    with db() as conn:
        rows = conn.execute('SELECT payload FROM projects').fetchall()
    return sorted([json.loads(row['payload']) for row in rows], key=lambda p:p['updated_at'], reverse=True)

def save_project(project):
    with db() as conn:
        conn.execute('INSERT OR REPLACE INTO projects VALUES (?,?)', (project['id'], json.dumps(project)))

def mutate_project(project_id, mutation):
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT payload FROM projects WHERE id=?', (project_id,)).fetchone()
        if not row:
            raise KeyError(project_id)
        project = json.loads(row['payload'])
        mutation(project)
        project['updated_at'] = now()
        conn.execute('UPDATE projects SET payload=? WHERE id=?', (json.dumps(project), project_id))
    return project

def public_project(project):
    result = json.loads(json.dumps(project))
    for photo in result['photos']:
        photo.pop('path', None)
        photo.pop('sha256', None)
    result['stale'] = bool(result.get('artifact') and (result['artifact']['revision'] != result['revision'] or result['artifact'].get('metadata',{}).get('engine_version') != ENGINE_VERSION))
    return result

def get_job(job_id):
    with db() as conn:
        row = conn.execute('SELECT payload FROM jobs WHERE id=?', (job_id,)).fetchone()
    return json.loads(row['payload']) if row else None

def save_job(job):
    with db() as conn:
        conn.execute('UPDATE jobs SET status=?,payload=? WHERE id=?', (job['status'], json.dumps(job), job['id']))
