# Product Requirement Document (PRD)
## Multi-View Image-to-3D Asset Generation Pipeline for Eyewear Virtual Try-On

## 1. Document Overview & Executive Summary
This document establishes the product requirements for a standalone, specialized Image-to-3D application. The core objective is to allow eyewear merchants to upload multiple flat 2D static product images (Front, Left Profile, Right Profile, Back, and a primary Hero shot) and instantly generate an industry-standard, symmetric, high-fidelity 3D asset (`.glb` / `.gltf`) optimized for real-time Virtual Try-On (VTO) engines. Generated assets are automatically hosted and stored on Cloudflare R2 storage.

---

## 2. The Core Problem & Architecture Justification
### The Flaw of Pure Generative 3D Models (Hunyuan3D V3 / TripoSR)
Using standard end-to-end 3D generation networks (like the proprietary Hunyuan3D V3 API or TripoSR) fails catastrophically for eyewear retail:
1. **Asymmetry:** Standard 3D networks generate organic, hand-drawn styles. Eyeglasses require sub-millimeter geometric symmetry; any slight variation in rim size or temple length renders the VTO alignment broken.
2. **Structural Distortion:** Thin, mechanical elements like temple arms, hinges, and bridge wires collapse into wobbly, clay-like structures when built by diffusion models.
3. **Material Blindness:** Pure 3D generative networks cannot isolate clear glass lenses from the opaque frame. They output glasses as solid, manifold chunks instead of transparent panes.

### The Necessity of the Multi-View Hybrid Deformable Pipeline with Self-Hosted Hunyuan3D-2
To achieve e-commerce grade fidelity, this application utilizes a **Hybrid Template Deformation + Generative Texture Fusion** pipeline using the open-source **Hunyuan3D-2** framework for the texture phase.

By upgrading the input to **Multi-View Uploads (Front, Left, Right, Back)**, we eliminate AI hallucination. The system does not have to guess what the sides look like; it uses Computer Vision to warp a mathematically perfect 3D template base to match the dimensions, then leverages our self-hosted **Hunyuan3D-2 / Hunyuan3D-Paint** texture engine solely to seam-blend, inpaint hidden crevices, and extract high-fidelity Physically Based Rendering (PBR) texture maps.

---

## 3. System Workflow Architecture
```unset
[Merchant Interface: 5-Image Upload Engine]
                     │
                     ▼
       [Stage 1: Multi-View Segmentation]
   (SAM 2 isolates backgrounds across all views)
                     │
                     ▼
    [Stage 2: Parametric Template Warping]
  (Frontal keypoints stretch a perfect 3D base mesh)
                     │
                     ▼
   [Stage 3: Multi-View Texture Projection]
(Front, Left, Right, Back colors projected onto mesh)
                     │
                     ▼
  [Stage 4: GENERATIVE AI SEAM BLENDING & PBR]
(Self-Hosted Hunyuan3D-2 Paint fills gaps + gloss maps)
                     │
                     ▼
        [Stage 5: R2 Upload & DB Entry]
```

### Detailed Pipeline Stages
1. **Stage 1: Multi-View Segmentation & Keypoints:** Segment Anything 2 (SAM 2) extracts the frame contours from all 5 images. YOLOv8-Pose detects precise 2D landmarks (hinge points, rim bounds, bridge width, temple curve).
2. **Stage 2: Parametric Template Warping:** The backend maps the frontal keypoints to an internal library of perfect 3D CAD frame templates (Aviator, Wayfarer, Round, etc.). The model expands, contracts, and depth-adjusts the symmetrical template using rigid deformation algorithms.
3. **Stage 3: Texture Projection:** Pixel colors from all four multi-view images are back-projected onto the corresponding faces of the warped 3D mesh via automated UV projection mapping.
4. **Stage 4: Generative AI Seam Blending & PBR Creation:** Our self-hosted **Hunyuan3D-2 / Hunyuan3D-Paint** execution stack inspects the boundaries where the photos overlap, seamlessly inpainting transition lines. Concurrently, it identifies the material (acetate vs. metal) and splits the output into separate PBR material layers:
   * **Albedo (Color):** Base pattern textures.
   * **Roughness/Metallic:** Controls reflection sheen on metal hinges vs plastic rims.
   * **Transmission/Opacity:** Permanently applied to the lens sub-mesh to keep it clear and transparent.
5. **Stage 5: Cloud Storage Pipeline:** The finished mesh is compiled into a highly-compressed, web-optimized `.glb` file and piped securely to **Cloudflare R2**.

---

## 4. Functional Specifications & User Interface (UI)
### 4.1 Multi-View Upload Interface
* **The Grid:** A structured drag-and-drop interface containing 5 explicit upload slots:
  1. **Primary/Hero Image** (used for asset reference thumbnail)
  2. **Front View** (perfectly flat face-on portrait)
  3. **Left Profile View** (temple arm and side rim profile)
  4. **Right Profile View** (temple arm and side rim profile)
  5. **Back View** (inside view showing the nose-pads and inner rim structure)
* **Overlay Guides:** Each upload slot must display a semi-transparent bounding-box silhouette overlay matching the expected camera angle. This forces the merchant to align the eyeglasses consistently to preserve relative scale.
* **Validation Check:** System blocks processing if any of the mandatory views (Front, Left, Right) are missing or fail raw edge-detection checks.

### 4.2 Real-Time Processing Status
* A progressive step-indicator tracking backend execution:
  * `[Step 1/4] Analyzing frame symmetry and isolating backgrounds...`
  * `[Step 2/4] Sculpting 3D geometry from multi-view coordinates...`
  * `[Step 3/4] Running self-hosted Hunyuan3D-2 texture pipeline...`
  * `[Step 4/4] Optimizing asset and storing files...`

### 4.3 Interactive 3D Interactive Previewer
* Upon completion, the UI dynamically mounts a lightweight WebGL renderer (Three.js or `<model-viewer>`).
* **Controls:** 360-degree orbit rotation, scroll-to-zoom, and a "Toggle Environment Lighting" dropdown to verify how the metallic/glossy materials react to diverse virtual lighting scenarios (Studio, Outdoor, Indoor).
* **Action Button:** A prominent "Save and Sync Asset" button that finalizes the Cloudflare R2 bucket state.

---

## 5. Technical Stack & Hardware Profiles
### 5.1 Backend Software Requirements
* **API Framework:** FastAPI (Python) for asynchronous multi-image streaming payload handling.
* **Segmentation Engine:** Meta’s **Segment Anything 2 (SAM 2)** (`sam2_hiera_large.pt`).
* **Computer Vision Processing:** OpenCV, PyTorch, and open3d (for point-cloud alignment and mesh optimization manipulation).
* **Texture / Generative Engine:** Self-hosted open-source **Tencent-Hunyuan/Hunyuan3D-2** weights and pipelines (`Hunyuan3D-Paint`).
* **Storage SDK:** Python `boto3` pointing to Cloudflare R2 S3-compatible API endpoints.

### 5.2 Infrastructure Hardware Specifications
Processing multiple high-resolution source images concurrently drastically increases VRAM overhead since the segmentation and diffusion texture passes are run across multiple perspectives.

#### Profile A: Minimum Environment (Local Machine / Dev Staging)
* **GPU:** 1x NVIDIA RTX 4070 Ti Super or RTX 3090 (**16GB VRAM**)
* **System RAM:** 32 GB System Memory
* **Storage:** 100 GB NVMe M.2 SSD (for quick template caching and local weight loading)
* *Pipeline Behavior:* To avoid Out-Of-Memory (OOM) fatal crashes, the backend will process the multi-view frames sequentially, queuing image data batches through the VRAM stack. Total pipeline generation time: **~90 to 120 seconds**.

#### Profile B: Recommended Production Profile (High-Throughput Enterprise SaaS)
* **GPU:** 1x NVIDIA RTX 4090 or NVIDIA A10G / L4 (**24GB VRAM**)
* **System RAM:** 64 GB System Memory
* **Network Interconnect:** 10 Gbps (for immediate streaming download/upload of multi-image matrix datasets)
* *Pipeline Behavior:* The system can simultaneously cache the full SAM 2 weights, the PyTorch template deformation variables, and the self-hosted Hunyuan3D-2 paint stack inside memory. It processes image profiles concurrently. Total pipeline generation time: **~25 to 45 seconds**.

---

## 6. Cloudflare R2 Storage Schema & Data Structure
Assets must be stored in a production-ready, globally distributed file hierarchy to prevent latency during checkout/try-on phases on merchant storefronts.

```unset
r2-bucket-eyewear-vto/
│
└─── assets/
     └─── merchant_id_77419/
          └─── product_sku_99302/
               ├─── frame_model.glb         <-- Highly compressed, web-optimized production asset
               ├─── thumbnail_hero.webp     <-- Processed, compressed display image
               ├─── raw_inputs/             <-- Archived original multi-view files
               │    ├─── front.png
               │    ├─── left.png
               │    ├─── right.png
               │    └─── back.png
               └─── metadata.json           <-- Dimensions data, structural scales, material settings
```