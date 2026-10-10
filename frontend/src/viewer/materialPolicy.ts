import * as THREE from 'three';

// Default anisotropy=1 loses printed detail on long arms viewed obliquely.
// Keep mipmaps for stability, but use the device-supported anisotropic filter.
export function preserveTextureDetail(root: THREE.Object3D, renderer: THREE.WebGLRenderer) {
  const maximum = Math.min(16, renderer.capabilities.getMaxAnisotropy());
  root.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      const texture = (material as THREE.MeshStandardMaterial).map;
      if (texture) { texture.anisotropy = maximum; texture.needsUpdate = true; }
    }
  });
}

// Frame and temple maps contain photographed color. Keep those maps unlit in
// inspection and AR. Solid hardware also bypasses relighting in this mode.
// Photo-estimated lenses use linear GLB factors; authored optical materials
// retain their GLB lighting policy. Alpha settings are never overwritten.
export function preservePhotoTextureColors(root: THREE.Object3D) {
  root.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return;
    const convert = (material: THREE.Material) => {
      if (material instanceof THREE.MeshBasicMaterial && material.name==='photo-estimated-lens') {
        material.toneMapped=false;
        if (material.map) material.alphaTest=.001;
        material.side=THREE.FrontSide;material.depthWrite=false;
        return material;
      }
      const optical = node.name.startsWith('lens_') || node.name === 'shield_lens';
      if (!(material instanceof THREE.MeshStandardMaterial) || (optical && !material.map && material.name !== 'photo-estimated-lens')) return material;
      const unlit = new THREE.MeshBasicMaterial({
        map: material.map,
        alphaMap: material.alphaMap,
        color: material.color,
        vertexColors: material.vertexColors,
        opacity: material.opacity,
        transparent: material.transparent,
        alphaTest: material.alphaTest,
        side: material.name === 'photo-estimated-lens' ? THREE.FrontSide : material.side,
        depthTest: material.depthTest,
        depthWrite: material.name === 'photo-estimated-lens' ? false : material.depthWrite,
        premultipliedAlpha: material.premultipliedAlpha,
        toneMapped: false,
      });
      material.dispose();
      return unlit;
    };
    node.material = Array.isArray(node.material) ? node.material.map(convert) : convert(node.material);
  });
}
