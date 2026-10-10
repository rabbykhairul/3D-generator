/** Approximate weak-perspective scale; mesh rotation supplies the visible yaw. */
export function pixelsPerMeter(eyePixels:number,pdMillimeters:number,yawRadians:number,sizePercent=100){
 if(!Number.isFinite(eyePixels)||!Number.isFinite(pdMillimeters)||pdMillimeters<=0)return 0;
 const foreshortening=Math.max(.55,Math.abs(Math.cos(yawRadians)));
 return eyePixels/foreshortening/(pdMillimeters/1000)*sizePercent/100;
}
