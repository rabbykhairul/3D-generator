# GHCR baked-model build

The workflow in `.github/workflows/build-and-push.yml` builds on pushes to `main` and on manual dispatch.

## Required repository configuration

Create these GitHub Actions repository variables:

- `YOLO_MODEL_REPO`: Hugging Face repository containing the custom pose weight.
- `YOLO_MODEL_REV`: immutable commit SHA in that repository.

Create this Actions secret when the model repository is private:

- `HF_TOKEN`: read-only Hugging Face token. It is exposed to one BuildKit secret mount and is not persisted in an image layer.

The expected file inside `YOLO_MODEL_REPO` is `eyewear_pose.pt`. A missing variable, inaccessible repository, or missing file intentionally fails the build.

The workflow uses `GITHUB_TOKEN` to push:

```text
ghcr.io/<owner>/<repository>:latest
ghcr.io/<owner>/<repository>:<commit-sha>
```

## Build behavior

The CPU-only test job never downloads model weights. The image job downloads pinned model revisions during the `model-bake` stage. Registry-backed BuildKit cache prevents unchanged model layers from being downloaded again on later builds.

Do not pass R2 credentials during build. Provide them only to the deployed serverless worker.

The merchant interface is compiled before the Python quality checks and is bundled into the same final image. If `R2_PUBLIC_BASE_URL` is on another origin, configure that R2/custom domain to allow browser `GET` requests from the application origin.
