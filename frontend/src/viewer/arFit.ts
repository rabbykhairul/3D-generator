export const defaultARFit = { pd: 63, scale: 100, horizontal: 0, vertical: 0, depth: 0, headWidth: 100, headHeight: 100, headHorizontal: 0, headVertical: 0, skullDepth: 0 };
export type ARFit = typeof defaultARFit;
const ranges: Record<keyof ARFit, readonly [number, number]> = {
  pd: [50, 75], scale: [70, 135], horizontal: [-80, 80], vertical: [-80, 80], depth: [-30, 30],
  headWidth: [80, 120], headHeight: [80, 120], headHorizontal: [-20, 20], headVertical: [-20, 20], skullDepth: [-30, 30],
};
export const fitKey = (modelUrl: string) => `ar-fit-v1:${modelUrl}`;

export function readARFit(modelUrl: string, storage?: Pick<Storage, 'getItem'>): ARFit {
  try {
    const data: unknown = JSON.parse((storage ?? localStorage).getItem(fitKey(modelUrl)) || '{}');
    if (!data || typeof data !== 'object') return { ...defaultARFit };
    const values = data as Record<string, unknown>;
    const fit = { ...defaultARFit };
    for (const key of Object.keys(fit) as (keyof ARFit)[]) {
      const value = values[key];
      const [min, max] = ranges[key];
      if (typeof value === 'number' && Number.isFinite(value)) fit[key] = Math.min(max, Math.max(min, value));
    }
    return fit;
  } catch { return { ...defaultARFit }; }
}

export function saveARFit(modelUrl: string, fit: ARFit, storage?: Pick<Storage, 'setItem'>) {
  try { (storage ?? localStorage).setItem(fitKey(modelUrl), JSON.stringify(fit)); } catch { /* Storage may be unavailable in private browsing. */ }
}
