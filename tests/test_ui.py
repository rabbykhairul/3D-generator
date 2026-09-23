from pathlib import Path

from fastapi.testclient import TestClient

from eyewear_vto.config import Settings
from eyewear_vto.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ui_source_contains_complete_capture_and_review_flow() -> None:
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "frontend" / "src" / "app.js").read_text(encoding="utf-8")

    assert "id=\"generation-form\"" in html
    assert "<model-viewer" in html
    assert "Save and sync asset" in html
    for view in ("front", "left", "right", "back", "hero"):
        assert f'id: "{view}"' in script
    assert 'fetch("/v1/jobs"' in script
    assert "/finalize" in script


def test_built_ui_and_local_assets_are_served(tmp_path: Path) -> None:
    ui_root = PROJECT_ROOT / "frontend" / "dist"
    settings = Settings(
        work_root=tmp_path / "work",
        local_storage_root=tmp_path / "storage",
        ui_root=ui_root,
        _env_file=None,
    )
    asset = settings.local_storage_root / "preview" / "frame.glb"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"glb")

    with TestClient(create_app(settings)) as client:
        root = client.get("/", follow_redirects=False)
        ui = client.get("/ui/")
        served_asset = client.get("/assets/preview/frame.glb")

    assert root.status_code == 307
    assert root.headers["location"] == "/ui/"
    assert ui.status_code == 200
    assert "Eyewear Asset Lab" in ui.text
    assert served_asset.content == b"glb"
