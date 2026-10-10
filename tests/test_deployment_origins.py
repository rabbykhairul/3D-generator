from fastapi.testclient import TestClient
from backend.main import app


def test_local_default_and_remote_exact_origin(monkeypatch):
    monkeypatch.delenv('EYEWEAR_ALLOWED_ORIGINS',raising=False)
    client=TestClient(app)
    assert client.get('/api/health',headers={'Origin':'http://localhost:5188'}).status_code==200
    assert client.get('/api/health',headers={'Origin':'https://evil.example'}).status_code==403
    monkeypatch.setenv('EYEWEAR_ALLOWED_ORIGINS','https://frames.example.com')
    for origin in ('https://frames.example.com.evil.test','http://frames.example.com','https://frames.example.com:444','null','http://localhost:5188'):
        assert client.get('/api/health',headers={'Origin':origin}).status_code==403
    assert client.get('/api/health',headers={'Origin':'https://frames.example.com'}).status_code==200
    assert client.get('/api/health').status_code==200
