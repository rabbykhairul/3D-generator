# TripoSR eyewear baseline

This is an isolated experiment using the [official TripoSR code](https://github.com/VAST-AI-Research/TripoSR)
and [official pretrained weights](https://huggingface.co/stabilityai/TripoSR).
It uses the collected Vincent Chase three-quarter photo by default, runs on
CPU, exports a GLB, and records basic mesh validity and elapsed time. It does
not replace the app's reconstruction pipeline or claim product fidelity.

From the repository root, with internet access:

```bash
bash scripts/setup_triposr_probe.sh
```

Output: `artifacts/triposr-baseline/0/mesh.glb`, `runner.log`, and `report.json`.
The first run downloads about 1.7 GB of weights and installs a separate Python
environment. The official default mesh grid is 256; the probe uses 128 for a
faster CPU feasibility check. CPU runtime may be substantially longer than the
paper's A100 GPU timing.

To try a different photo after setup:

```bash
artifacts/triposr-venv/bin/python scripts/run_triposr_probe.py \
  --image /absolute/path/to/product-photo.webp \
  --output artifacts/triposr-other-product
```

Before treating the output as a product asset, compare front, left and right
renders with the corresponding source photos. Check separated lens openings,
arm roots and tips, hinges, nose pads, frame and lens thickness, tint and
transparency. One input image cannot establish hidden depth or dimensions.

In the current execution sandbox, GitHub and Hugging Face downloads fail DNS
resolution, and the official weights are not cached. The probe's `report.json`
therefore says `blocked`; **no TripoSR GLB was generated here**.
