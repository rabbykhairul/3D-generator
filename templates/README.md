# Production CAD template contract

`catalog.json` follows `catalog.example.json`. Every referenced GLB must:

- use metre units with the physical dimensions recorded in millimetres in the catalog;
- use `+Y` up and `+Z` forward;
- place the origin at the bridge centre;
- preserve the six exact semantic node names listed in the catalog;
- provide non-overlapping UVs;
- keep left and right lenses as separate meshes;
- keep hinge pivots and open temple geometry symmetric;
- contain no external buffers or image URIs.

The deformation JSON is template-specific. It must have `schema_version: 1` and a non-empty `anchors` mapping from canonical landmark names to mesh handles; it may also define symmetry pairs and locked regions. Production deformation cannot be completed until at least one reviewed GLB and its handle map are supplied.

Templates are versioned artifacts and should be stored in the baked image under `/opt/templates`, not downloaded when the worker starts.

The final Docker build deliberately fails until `catalog.json`, every referenced GLB, and every referenced deformation map are present and structurally valid.
