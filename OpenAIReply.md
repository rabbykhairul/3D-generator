You absolutely can build this without any OpenAI vision model. For your sunglasses → component detection → 3D pipeline, I’d actually recommend a **specialized computer-vision stack** if you want predictable segmentation.

A strong architecture would be:

```text
Product image
     │
     ▼
OpenCV preprocessing
     │
     ▼
Part detector
     │
     ├── Option A: Grounding DINO
     └── Option B: trained YOLO segmentation model
     │
     ▼
SAM 2 segmentation
     │
     ▼
Pixel masks
     │
     ▼
OpenCV contours
     │
     ▼
Polygon / SVG / JSON
     │
     ▼
Blender Python
     │
     ▼
GLB / OBJ / FBX
```

For a prototype, I would use **Grounding DINO + SAM 2**. Grounding DINO receives prompts such as:

```text
left lens
right lens
bridge
left nose pad
right nose pad
left hinge
right hinge
left temple arm
right temple arm
```

and gives approximate bounding boxes. Then SAM 2 produces the actual masks.

Conceptually:

```python
image = load_image("glasses.jpg")

boxes = grounding_dino.detect(
    image,
    text_prompt=
    "lens . bridge . nose pad . hinge . temple arm . temple tip"
)

masks = sam2.segment(
    image,
    boxes=boxes
)
```

Then OpenCV converts each mask into polygons:

```python
import cv2

contours, _ = cv2.findContours(
    mask.astype("uint8"),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = max(contours, key=cv2.contourArea)

epsilon = 0.002 * cv2.arcLength(contour, True)

polygon = cv2.approxPolyDP(
    contour,
    epsilon,
    True
)
```

Your output could then be:

```json
{
  "parts": {
    "lens_left": {
      "class": "lens",
      "polygon": [
        [110, 380],
        [185, 355],
        [290, 350],
        [390, 382]
      ]
    },

    "bridge": {
      "class": "bridge",
      "polygon": [
        [478, 460],
        [510, 450],
        [550, 452]
      ]
    },

    "nose_pad_left": {
      "class": "nose_pad",
      "polygon": []
    }
  }
}
```

### But there's an even better option for your use case

If your application is specifically for **eyeglasses/sunglasses**, I would eventually remove Grounding DINO too.

Train a custom instance-segmentation model with fixed eyewear classes:

```text
0 lens_left
1 lens_right
2 rim_left
3 rim_right
4 bridge
5 nose_pad_left
6 nose_pad_right
7 pad_arm_left
8 pad_arm_right
9 hinge_left
10 hinge_right
11 endpiece_left
12 endpiece_right
13 temple_left
14 temple_right
15 temple_tip_left
16 temple_tip_right
17 decorative_part_left
18 decorative_part_right
```

Then your pipeline becomes much simpler:

```text
           IMAGE
             │
             ▼
       YOLO-Seg model
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     lens  bridge temple
      mask   mask   mask
       │      │      │
       └──────┼──────┘
              ▼
            OpenCV
              ▼
      Contours / polygons
              ▼
          Blender
```

For production, this would be my preference.

You could use an Ultralytics segmentation model, for example:

```bash
pip install ultralytics opencv-python numpy
```

Dataset layout:

```text
eyewear_dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

And a configuration like:

```yaml
path: eyewear_dataset

train: images/train
val: images/val

names:
  0: lens_left
  1: lens_right
  2: bridge
  3: nose_pad_left
  4: nose_pad_right
  5: hinge_left
  6: hinge_right
  7: temple_left
  8: temple_right
  9: temple_tip_left
  10: temple_tip_right
```

Training is roughly:

```python
from ultralytics import YOLO

model = YOLO("yolo-seg-model.pt")

model.train(
    data="eyewear.yaml",
    epochs=100,
    imgsz=1024,
    batch=8
)
```

Then inference:

```python
result = model("sunglasses.jpg")[0]

for mask, cls in zip(
    result.masks.xy,
    result.boxes.cls
):
    class_id = int(cls)

    print(
        model.names[class_id],
        mask
    )
```

`mask` can already contain polygon coordinates, which is extremely convenient for what you're trying to do.

### The tricky part: transparent lenses

Your sunglasses images expose one of the harder computer-vision cases:

```text
lens
 ├── transparent
 ├── gradient
 ├── temple visible through lens
 ├── reflections
 └── sometimes no physical rim
```

For example, the rimless pair you showed has effectively:

```text
        lens boundary
       ╱             ╲
      ╱               ╲
   ●────────────────────●
  hinge              bridge
```

There isn't a conventional rim for a segmentation model to follow.

Generic SAM can sometimes mistake the temple visible **through the lens** for part of the lens boundary.

This is one reason a custom eyewear segmentation model can outperform generic segmentation once you have enough labeled images.

### You also don't necessarily need AI for left/right classification

Suppose the model detects two masks classified simply as `lens`.

You can determine left/right geometrically:

```python
centers = []

for polygon in lens_polygons:
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()

    centers.append((cx, cy))

centers.sort(key=lambda p: p[0])

lens_left = centers[0]
lens_right = centers[1]
```

Same idea for:

```text
nose pads
hinges
temples
temple tips
decorations
```

So the neural model doesn't have to learn:

```text
left_lens
right_lens
```

It could simply detect:

```text
lens
```

and your code assigns left/right afterward. That's usually a cleaner dataset design.

### Another useful architecture

I'd separate **semantic identity** from **geometry extraction**:

```text
                  INPUT IMAGE
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Detector              SAM 2
     "what is it?"        "which pixels?"
             │                   │
             └─────────┬─────────┘
                       ▼
                  COMPONENT
                     MASKS
                       │
                       ▼
                    OpenCV
                       │
              ┌────────┴─────────┐
              ▼                  ▼
           contours           keypoints
              │                  │
              └────────┬─────────┘
                       ▼
                    JSON
                       │
                       ▼
                Blender script
```

### For the 3D side, I'd use more than segmentation

From the front image:

```text
X = horizontal shape
Z = vertical shape
```

From a side image:

```text
Y = depth
```

So:

```text
Front mask
    +
Side mask
    +
known component topology
    ↓
3D parameter estimation
```

For eyewear, I wouldn't rely entirely on generic image-to-3D models. Glasses are mostly **parametric objects**, so you can generate cleaner geometry with rules.

For instance:

```text
LENS
polygon → smooth spline
        → slight curvature
        → solidify 1.5–2 mm

RIM
lens boundary
        → curve
        → bevel profile

BRIDGE
centerline
        → Bézier curve
        → circular bevel

TEMPLE
detected centerline
        → 3D curve
        → bevel

NOSE PAD
template primitive
        → scale
        → position from detection
```

That gives you editable CAD-like geometry rather than an unpredictable AI mesh.

### Stack I'd personally choose

For your project:

| Purpose | Technology |
|---|---|
| Image loading | OpenCV |
| Initial automatic annotation | Grounding DINO |
| Segmentation | SAM 2 |
| Production component detector | custom YOLO segmentation |
| Polygon processing | OpenCV |
| Geometry math | NumPy |
| Curves/polygon cleanup | Shapely |
| Dataset annotation | CVAT or Label Studio |
| 3D generation | Blender Python API |
| Mesh manipulation | trimesh |
| Output | GLB/GLTF |

Once you have around **hundreds to a few thousand well-annotated eyewear photos**, I would make the custom segmentation model the primary detector and use SAM only to speed up annotation or correct difficult examples.

So you can make this entire system **without OpenAI, without ChatGPT, and potentially completely offline**:

```text
Python
  +
PyTorch
  +
YOLO-Seg
  +
SAM 2
  +
OpenCV
  +
Blender
```

For the particular type of system you're building, that may actually be the better long-term architecture.




########### ASKED for the accuracy and got the reply

You absolutely can build this without any OpenAI vision model. For your sunglasses → component detection → 3D pipeline, I’d actually recommend a **specialized computer-vision stack** if you want predictable segmentation.

A strong architecture would be:

```text
Product image
     │
     ▼
OpenCV preprocessing
     │
     ▼
Part detector
     │
     ├── Option A: Grounding DINO
     └── Option B: trained YOLO segmentation model
     │
     ▼
SAM 2 segmentation
     │
     ▼
Pixel masks
     │
     ▼
OpenCV contours
     │
     ▼
Polygon / SVG / JSON
     │
     ▼
Blender Python
     │
     ▼
GLB / OBJ / FBX
```

For a prototype, I would use **Grounding DINO + SAM 2**. Grounding DINO receives prompts such as:

```text
left lens
right lens
bridge
left nose pad
right nose pad
left hinge
right hinge
left temple arm
right temple arm
```

and gives approximate bounding boxes. Then SAM 2 produces the actual masks.

Conceptually:

```python
image = load_image("glasses.jpg")

boxes = grounding_dino.detect(
    image,
    text_prompt=
    "lens . bridge . nose pad . hinge . temple arm . temple tip"
)

masks = sam2.segment(
    image,
    boxes=boxes
)
```

Then OpenCV converts each mask into polygons:

```python
import cv2

contours, _ = cv2.findContours(
    mask.astype("uint8"),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = max(contours, key=cv2.contourArea)

epsilon = 0.002 * cv2.arcLength(contour, True)

polygon = cv2.approxPolyDP(
    contour,
    epsilon,
    True
)
```

Your output could then be:

```json
{
  "parts": {
    "lens_left": {
      "class": "lens",
      "polygon": [
        [110, 380],
        [185, 355],
        [290, 350],
        [390, 382]
      ]
    },

    "bridge": {
      "class": "bridge",
      "polygon": [
        [478, 460],
        [510, 450],
        [550, 452]
      ]
    },

    "nose_pad_left": {
      "class": "nose_pad",
      "polygon": []
    }
  }
}
```

### But there's an even better option for your use case

If your application is specifically for **eyeglasses/sunglasses**, I would eventually remove Grounding DINO too.

Train a custom instance-segmentation model with fixed eyewear classes:

```text
0 lens_left
1 lens_right
2 rim_left
3 rim_right
4 bridge
5 nose_pad_left
6 nose_pad_right
7 pad_arm_left
8 pad_arm_right
9 hinge_left
10 hinge_right
11 endpiece_left
12 endpiece_right
13 temple_left
14 temple_right
15 temple_tip_left
16 temple_tip_right
17 decorative_part_left
18 decorative_part_right
```

Then your pipeline becomes much simpler:

```text
           IMAGE
             │
             ▼
       YOLO-Seg model
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     lens  bridge temple
      mask   mask   mask
       │      │      │
       └──────┼──────┘
              ▼
            OpenCV
              ▼
      Contours / polygons
              ▼
          Blender
```

For production, this would be my preference.

You could use an Ultralytics segmentation model, for example:

```bash
pip install ultralytics opencv-python numpy
```

Dataset layout:

```text
eyewear_dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

And a configuration like:

```yaml
path: eyewear_dataset

train: images/train
val: images/val

names:
  0: lens_left
  1: lens_right
  2: bridge
  3: nose_pad_left
  4: nose_pad_right
  5: hinge_left
  6: hinge_right
  7: temple_left
  8: temple_right
  9: temple_tip_left
  10: temple_tip_right
```

Training is roughly:

```python
from ultralytics import YOLO

model = YOLO("yolo-seg-model.pt")

model.train(
    data="eyewear.yaml",
    epochs=100,
    imgsz=1024,
    batch=8
)
```

Then inference:

```python
result = model("sunglasses.jpg")[0]

for mask, cls in zip(
    result.masks.xy,
    result.boxes.cls
):
    class_id = int(cls)

    print(
        model.names[class_id],
        mask
    )
```

`mask` can already contain polygon coordinates, which is extremely convenient for what you're trying to do.

### The tricky part: transparent lenses

Your sunglasses images expose one of the harder computer-vision cases:

```text
lens
 ├── transparent
 ├── gradient
 ├── temple visible through lens
 ├── reflections
 └── sometimes no physical rim
```

For example, the rimless pair you showed has effectively:

```text
        lens boundary
       ╱             ╲
      ╱               ╲
   ●────────────────────●
  hinge              bridge
```

There isn't a conventional rim for a segmentation model to follow.

Generic SAM can sometimes mistake the temple visible **through the lens** for part of the lens boundary.

This is one reason a custom eyewear segmentation model can outperform generic segmentation once you have enough labeled images.

### You also don't necessarily need AI for left/right classification

Suppose the model detects two masks classified simply as `lens`.

You can determine left/right geometrically:

```python
centers = []

for polygon in lens_polygons:
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()

    centers.append((cx, cy))

centers.sort(key=lambda p: p[0])

lens_left = centers[0]
lens_right = centers[1]
```

Same idea for:

```text
nose pads
hinges
temples
temple tips
decorations
```

So the neural model doesn't have to learn:

```text
left_lens
right_lens
```

It could simply detect:

```text
lens
```

and your code assigns left/right afterward. That's usually a cleaner dataset design.

### Another useful architecture

I'd separate **semantic identity** from **geometry extraction**:

```text
                  INPUT IMAGE
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Detector              SAM 2
     "what is it?"        "which pixels?"
             │                   │
             └─────────┬─────────┘
                       ▼
                  COMPONENT
                     MASKS
                       │
                       ▼
                    OpenCV
                       │
              ┌────────┴─────────┐
              ▼                  ▼
           contours           keypoints
              │                  │
              └────────┬─────────┘
                       ▼
                    JSON
                       │
                       ▼
                Blender script
```

### For the 3D side, I'd use more than segmentation

From the front image:

```text
X = horizontal shape
Z = vertical shape
```

From a side image:

```text
Y = depth
```

So:

```text
Front mask
    +
Side mask
    +
known component topology
    ↓
3D parameter estimation
```

For eyewear, I wouldn't rely entirely on generic image-to-3D models. Glasses are mostly **parametric objects**, so you can generate cleaner geometry with rules.

For instance:

```text
LENS
polygon → smooth spline
        → slight curvature
        → solidify 1.5–2 mm

RIM
lens boundary
        → curve
        → bevel profile

BRIDGE
centerline
        → Bézier curve
        → circular bevel

TEMPLE
detected centerline
        → 3D curve
        → bevel

NOSE PAD
template primitive
        → scale
        → position from detection
```

That gives you editable CAD-like geometry rather than an unpredictable AI mesh.

### Stack I'd personally choose

For your project:

| Purpose | Technology |
|---|---|
| Image loading | OpenCV |
| Initial automatic annotation | Grounding DINO |
| Segmentation | SAM 2 |
| Production component detector | custom YOLO segmentation |
| Polygon processing | OpenCV |
| Geometry math | NumPy |
| Curves/polygon cleanup | Shapely |
| Dataset annotation | CVAT or Label Studio |
| 3D generation | Blender Python API |
| Mesh manipulation | trimesh |
| Output | GLB/GLTF |

Once you have around **hundreds to a few thousand well-annotated eyewear photos**, I would make the custom segmentation model the primary detector and use SAM only to speed up annotation or correct difficult examples.

So you can make this entire system **without OpenAI, without ChatGPT, and potentially completely offline**:

```text
Python
  +
PyTorch
  +
YOLO-Seg
  +
SAM 2
  +
OpenCV
  +
Blender
```

For the particular type of system you're building, that may actually be the better long-term architecture.




@