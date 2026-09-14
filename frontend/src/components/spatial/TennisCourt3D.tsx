import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Component, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ArrowsOut, ArrowCounterClockwise, Minus, Plus } from "@phosphor-icons/react";
import type { AnalysisRun } from "../../data/contracts";
import { COURT, PLAYER_COLORS, courtToWorld, densityGrid, distance, interpolatePosition, samplePoints, splitTrail, type CourtPoint, type FrameRange, type NormalizedTelemetry, type PlayerFilter } from "../../lib/tennisTelemetry";

type CameraAngle = "Tactical" | "Broadcast" | "Top" | "Cinematic";
const CAMERA_ANGLES: Record<CameraAngle, [number, number, number]> = { Tactical: [18, 24, 22], Broadcast: [0, 12, 31], Top: [0, 39, 0.01], Cinematic: [23, 16, 22] };
type SceneProps = { run: AnalysisRun; telemetry: NormalizedTelemetry; playhead: RefObject<number>; frame?: number; playing?: boolean; trails: boolean; arc: boolean; reducedMotion: boolean; density?: { filter: PlayerFilter; range: FrameRange } };

class SceneBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? this.props.fallback : this.props.children; }
}

export default function TennisCourt3D(props: SceneProps & { onFallback: () => void }) {
  const [angle, setAngle] = useState<CameraAngle>("Cinematic");
  const [revision, setRevision] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [lost, setLost] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const fallback = <div className="spatial-empty" role="status"><ArrowsOut size={34} /><h3>3D rendering unavailable</h3><p>This browser could not create a WebGL scene. Your telemetry is available in the tactical view.</p><button type="button" className="astra-button" onClick={props.onFallback}>Open 2D tactical view</button></div>;
  // Avoid probing a context here: Canvas owns and releases the only WebGL context.
  const supported = typeof window.WebGL2RenderingContext !== "undefined";
  return <div className="court3d-host" ref={root}>
    {!supported || lost ? fallback : <SceneBoundary fallback={fallback}>
      <Canvas shadows="percentage" frameloop="demand" dpr={[1, 1.6]} camera={{ position: CAMERA_ANGLES.Cinematic, fov: 43, near: 0.1, far: 150 }} gl={{ antialias: true, powerPreference: "high-performance" }} fallback={<span aria-hidden="true" />}
        onCreated={({ gl }) => { gl.domElement.setAttribute("aria-label", "Interactive 3D tennis court; use camera buttons to change perspective"); }}>
        <ReplayRenderDriver frame={props.frame ?? 0} playing={props.playing ?? false} />
        <ContextMonitor onLost={() => setLost(true)} />
        <color attach="background" args={["#111a1c"]} />
        <fog attach="fog" args={["#111a1c", 48, 95]} />
        <ambientLight intensity={0.75} />
        <hemisphereLight args={["#d9f3ff", "#284334", 1.3]} />
        <directionalLight castShadow position={[-10, 24, 4]} intensity={2.8} color="#fff9e8" shadow-mapSize={[1024, 1024]} shadow-camera-left={-22} shadow-camera-right={22} shadow-camera-top={24} shadow-camera-bottom={-24} shadow-normalBias={0.03} />
        <directionalLight position={[16, 12, -18]} intensity={1.4} color="#91d9de" />
        <CourtSurface />
        <CameraRig angle={angle} revision={revision} zoom={zoom} reducedMotion={props.reducedMotion} />
        {props.density ? <DensityColumns telemetry={props.telemetry} {...props.density} /> : <>
          <PlayerAvatar3D index={0} {...props} />
          <PlayerAvatar3D index={1} {...props} />
          <BallTrajectory3D {...props} />
          {props.trails ? <>
            <MovementTrail points={props.telemetry.players[0] ?? []} playhead={props.playhead} color={PLAYER_COLORS[0]} maxJump={1} />
            <MovementTrail points={props.telemetry.players[1] ?? []} playhead={props.playhead} color={PLAYER_COLORS[1]} maxJump={1} />
            <MovementTrail points={props.telemetry.ballPositions} playhead={props.playhead} color="#faf5ce" maxJump={5} count={28} />
          </> : null}
          <BounceMarkers run={props.run} playhead={props.playhead} />
        </>}
      </Canvas>
    </SceneBoundary>}
    <div className="court-camera-bar" role="group" aria-label="Camera angle">
      {(Object.keys(CAMERA_ANGLES) as CameraAngle[]).map(value => <button key={value} type="button" aria-pressed={angle === value} onClick={() => { setAngle(value); setZoom(1); setRevision(v => v + 1); }}>{value}</button>)}
    </div>
    <div className="court-tools" role="group" aria-label="3D camera controls">
      <button type="button" aria-label="Zoom in" onClick={() => setZoom(v => Math.max(0.65, v - 0.12))}><Plus size={17} /></button>
      <button type="button" aria-label="Zoom out" onClick={() => setZoom(v => Math.min(1.45, v + 0.12))}><Minus size={17} /></button>
      <button type="button" aria-label="Reset camera" onClick={() => { setZoom(1); setRevision(v => v + 1); }}><ArrowCounterClockwise size={17} /></button>
    </div>
    <div className="court-interaction-hint">Drag to orbit <span>·</span> Scroll to zoom</div>
    <div className="court-axis" aria-hidden="true"><span>Z</span><i /><b>X</b></div>
  </div>;
}

function ReplayRenderDriver({ frame, playing }: { frame: number; playing: boolean }) {
  const invalidate = useThree(state => state.invalidate);
  useEffect(() => invalidate(), [frame, playing, invalidate]);
  useFrame(() => { if (playing) invalidate(); });
  return null;
}

function ContextMonitor({ onLost }: { onLost: () => void }) {
  const { gl } = useThree();
  useEffect(() => {
    const canvas = gl.domElement;
    const handle = (event: Event) => { event.preventDefault(); onLost(); };
    canvas.addEventListener("webglcontextlost", handle);
    return () => canvas.removeEventListener("webglcontextlost", handle);
  }, [gl, onLost]);
  return null;
}

function CameraRig({ angle, revision, zoom, reducedMotion }: { angle: CameraAngle; revision: number; zoom: number; reducedMotion: boolean }) {
  const { camera, gl, invalidate } = useThree();
  const controls = useRef<OrbitControls | null>(null);
  const target = useRef(new THREE.Vector3(...CAMERA_ANGLES.Cinematic));
  const moving = useRef(true);
  useEffect(() => {
    const orbit = new OrbitControls(camera, gl.domElement);
    orbit.enableDamping = !reducedMotion;
    orbit.dampingFactor = 0.08;
    orbit.enablePan = false;
    orbit.minDistance = 13;
    orbit.maxDistance = 64;
    orbit.minPolarAngle = 0.001;
    orbit.maxPolarAngle = Math.PI / 2.1;
    orbit.target.set(0, 0, 0);
    const start = () => { moving.current = false; };
    const change = () => invalidate();
    orbit.addEventListener("start", start);
    orbit.addEventListener("change", change);
    controls.current = orbit;
    return () => { orbit.removeEventListener("start", start); orbit.removeEventListener("change", change); orbit.dispose(); };
  }, [camera, gl, invalidate, reducedMotion]);
  useEffect(() => { target.current.set(...CAMERA_ANGLES[angle]).multiplyScalar(zoom); moving.current = true; invalidate(); }, [angle, revision, zoom, invalidate]);
  useFrame((_, delta) => {
    if (moving.current) {
      camera.position.lerp(target.current, reducedMotion ? 1 : 1 - Math.exp(-5 * delta));
      if (camera.position.distanceTo(target.current) < 0.02) moving.current = false;
      else invalidate();
    }
    controls.current?.update();
  });
  return null;
}

function CourtSurface() {
  const { width: w, length: l, alley, service } = COURT;
  const lines = useMemo(() => {
    const result: number[] = [];
    const add = (x1: number, z1: number, x2: number, z2: number) => result.push(x1, 0.025, z1, x2, 0.025, z2);
    for (const x of [-w / 2, -w / 2 + alley, w / 2 - alley, w / 2]) add(x, -l / 2, x, l / 2);
    for (const z of [-l / 2, l / 2]) add(-w / 2, z, w / 2, z);
    for (const z of [-service, service]) add(-COURT.singles / 2, z, COURT.singles / 2, z);
    add(0, -service, 0, service);
    add(0, -l / 2, 0, -l / 2 + 0.15); add(0, l / 2, 0, l / 2 - 0.15);
    return new Float32Array(result);
  }, [alley, l, service, w]);
  const net = useMemo(() => {
    const points: number[] = [];
    const half = COURT.width / 2 + 0.914;
    for (let x = -half; x <= half; x += 0.19) {
      const height = COURT.netCenter + (COURT.netPost - COURT.netCenter) * (x / half) ** 2;
      points.push(x, 0.07, 0, x, height, 0);
    }
    for (let y = 0.1; y < COURT.netCenter; y += 0.12) points.push(-half, y, 0, half, y, 0);
    return new Float32Array(points);
  }, []);
  const tape = useMemo(() => {
    const half = COURT.width / 2 + 0.914;
    return new THREE.CatmullRomCurve3(Array.from({ length: 25 }, (_, i) => { const x = (i / 24 * 2 - 1) * half; return new THREE.Vector3(x, COURT.netCenter + (COURT.netPost - COURT.netCenter) * (x / half) ** 2, 0); }));
  }, []);
  return <group>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.22, 0]} receiveShadow><planeGeometry args={[160, 160]} /><meshStandardMaterial color="#101b1d" roughness={1} /></mesh>
    <gridHelper args={[120, 80, "#263837", "#1c2e2f"]} position={[0, -0.21, 0]} />
    <mesh position={[0, -0.13, 0]} receiveShadow><boxGeometry args={[18, 0.22, 34]} /><meshStandardMaterial color="#263e40" roughness={0.95} /></mesh>
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[w, l]} /><meshStandardMaterial color="#345c57" roughness={0.88} /></mesh>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.006, 0]} receiveShadow><planeGeometry args={[COURT.singles, service * 2]} /><meshStandardMaterial color="#3f6b63" roughness={0.9} /></mesh>
    <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[lines, 3]} /></bufferGeometry><lineBasicMaterial color="#e2eae0" transparent opacity={0.95} /></lineSegments>
    <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[net, 3]} /></bufferGeometry><lineBasicMaterial color="#b9c8bf" transparent opacity={0.4} /></lineSegments>
    <mesh castShadow><tubeGeometry args={[tape, 32, 0.035, 5, false]} /><meshStandardMaterial color="#eceedd" /></mesh>
    {[-1, 1].map(side => <group key={side} position={[side * (w / 2 + 0.914), 0, 0]}><mesh castShadow position={[0, COURT.netPost / 2, 0]}><cylinderGeometry args={[0.065, 0.075, COURT.netPost, 10]} /><meshStandardMaterial color="#c6d2ca" metalness={0.6} roughness={0.3} /></mesh><mesh position={[0, 0.025, 0]}><cylinderGeometry args={[0.22, 0.22, 0.05, 16]} /><meshStandardMaterial color="#142525" /></mesh></group>)}
    <mesh position={[0, COURT.netCenter / 2, 0.022]}><boxGeometry args={[0.045, COURT.netCenter, 0.01]} /><meshStandardMaterial color="#dce4d9" /></mesh>
    {[-1, 1].map(side => <mesh key={side} rotation={[-Math.PI / 2, 0, 0]} position={[side * 8.6, 0.005, 0]}><planeGeometry args={[0.035, 32]} /><meshBasicMaterial color="#5e8371" /></mesh>)}
  </group>;
}

function PlayerAvatar3D({ index, telemetry, playhead, reducedMotion }: SceneProps & { index: number }) {
  const group = useRef<THREE.Group>(null);
  const leftLeg = useRef<THREE.Group>(null); const rightLeg = useRef<THREE.Group>(null);
  const arm = useRef<THREE.Group>(null);
  const points = telemetry.players[index] ?? [];
  const color = PLAYER_COLORS[index]!;
  useFrame(() => {
    const avatar = group.current;
    if (!avatar) return;
    const frame = playhead.current;
    const point = interpolatePosition(points, frame);
    avatar.visible = !!point;
    if (!point) return;
    avatar.position.set(...courtToWorld(point));
    const before = points[Math.max(0, Math.floor(frame) - 1)];
    const speed = before ? Math.min(1, distance(before, point) * telemetry.fps / 6) : 0;
    const ball = interpolatePosition(telemetry.ballPositions, frame, 5);
    if (ball) avatar.rotation.y = Math.atan2(ball[0] - point[0], ball[1] - point[1]);
    else avatar.rotation.y = index === 0 ? Math.PI : 0;
    const gait = reducedMotion ? 0 : Math.sin(frame / telemetry.fps * 13) * speed * 0.55;
    if (leftLeg.current) leftLeg.current.rotation.x = gait;
    if (rightLeg.current) rightLeg.current.rotation.x = -gait;
    if (arm.current) arm.current.rotation.x = -0.25 - gait * 0.5;
  });
  return <group ref={group}>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}><ringGeometry args={[0.48, 0.56, 40]} /><meshBasicMaterial color={color} transparent opacity={0.8} /></mesh>
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.028, 0]}><circleGeometry args={[0.47, 32]} /><meshBasicMaterial color={color} transparent opacity={0.07} /></mesh>
    <mesh castShadow position={[0, 1.2, 0]} scale={[1, 1, 0.7]}><capsuleGeometry args={[0.23, 0.38, 5, 12]} /><meshStandardMaterial color={color} roughness={0.55} /></mesh>
    <mesh castShadow position={[0, 1.73, 0.01]}><sphereGeometry args={[0.16, 16, 12]} /><meshStandardMaterial color="#e1d2b5" roughness={0.8} /></mesh>
    <mesh position={[0, 1.8, 0.06]} scale={[1, 0.6, 1]}><sphereGeometry args={[0.17, 12, 8]} /><meshStandardMaterial color="#f3f0dc" /></mesh>
    <mesh position={[0, 0.87, 0]} scale={[1, 0.7, 0.85]} castShadow><sphereGeometry args={[0.25, 12, 8]} /><meshStandardMaterial color="#1e3034" /></mesh>
    {[-1, 1].map(side => <group key={side} ref={side === -1 ? leftLeg : rightLeg} position={[side * 0.14, 0.88, 0]} rotation={[0, 0, side * -0.12]}>
      <mesh position={[0, -0.26, 0]} castShadow><capsuleGeometry args={[0.085, 0.3, 4, 8]} /><meshStandardMaterial color="#bcc3b1" /></mesh>
      <mesh position={[0, -0.64, 0.055]} castShadow><capsuleGeometry args={[0.065, 0.27, 4, 8]} /><meshStandardMaterial color="#f2efdc" /></mesh>
      <mesh position={[0, -0.8, 0.12]} scale={[1, 0.65, 1.6]} castShadow><sphereGeometry args={[0.11, 10, 8]} /><meshStandardMaterial color="#e5e9da" /></mesh>
    </group>)}
    {[-1, 1].map(side => <group key={side} ref={side === 1 ? arm : undefined} position={[side * 0.26, 1.43, 0]} rotation={[-0.35, 0, side * 0.28]}>
      <mesh position={[0, -0.23, 0]} castShadow><capsuleGeometry args={[0.07, 0.31, 4, 8]} /><meshStandardMaterial color={color} /></mesh>
      <mesh position={[0, -0.45, 0.13]} rotation={[-0.9, 0, 0]}><capsuleGeometry args={[0.055, 0.23, 4, 8]} /><meshStandardMaterial color="#dfd0b1" /></mesh>
      {side === 1 ? <group position={[0, -0.56, 0.42]} rotation={[0.5, 0, 0]}>
        <mesh position={[0, -0.05, 0]}><cylinderGeometry args={[0.025, 0.025, 0.3, 6]} /><meshStandardMaterial color="#e7e4d6" /></mesh>
        <mesh position={[0, 0.24, 0]} scale={[0.8, 1, 1]}><torusGeometry args={[0.23, 0.024, 6, 24]} /><meshStandardMaterial color={color} /></mesh>
        <mesh position={[0, 0.24, 0]} scale={[0.8, 1, 1]}><circleGeometry args={[0.21, 24]} /><meshStandardMaterial color="#bccdc5" transparent opacity={0.3} side={THREE.DoubleSide} wireframe /></mesh>
      </group> : null}
    </group>)}
    <PlayerLabel text={`P${index + 1}`} color={color} />
  </group>;
}

function PlayerLabel({ text, color }: { text: string; color: string }) {
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas"); canvas.width = 128; canvas.height = 64;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#132023"; context.beginPath(); context.roundRect(4, 4, 120, 56, 14); context.fill();
    context.fillStyle = color; context.font = "bold 34px sans-serif"; context.textAlign = "center"; context.fillText(text, 64, 44);
    const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace; return map;
  }, [color, text]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <sprite position={[0, 2.38, 0]} scale={[1.05, 0.525, 1]}><spriteMaterial map={texture} transparent depthTest={false} /></sprite>;
}

function MovementTrail({ points, playhead, color, maxJump, count = 60 }: { points: ReadonlyArray<CourtPoint | null>; playhead: RefObject<number>; color: string; maxJump: number; count?: number }) {
  const positions = useMemo(() => new Float32Array(count * 6), [count]);
  const geometry = useRef<THREE.BufferGeometry>(null);
  const lastFrame = useRef(-1);
  useFrame(() => {
    const frame = Math.floor(playhead.current);
    if (!geometry.current || frame === lastFrame.current) return;
    lastFrame.current = frame;
    let offset = 0;
    for (const path of splitTrail(points, frame, count, maxJump)) for (let i = 1; i < path.length; i++) {
      positions.set(courtToWorld(path[i - 1]!, 0.055), offset); positions.set(courtToWorld(path[i]!, 0.055), offset + 3); offset += 6;
    }
    geometry.current.setDrawRange(0, offset / 3);
    geometry.current.attributes.position!.needsUpdate = true;
    geometry.current.computeBoundingSphere();
  });
  return <lineSegments frustumCulled={false}><bufferGeometry ref={geometry}><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry><lineBasicMaterial color={color} transparent opacity={0.7} /></lineSegments>;
}

function BallTrajectory3D({ telemetry, playhead, arc, run }: SceneProps) {
  const group = useRef<THREE.Group>(null);
  const shadow = useRef<THREE.Mesh>(null);
  const events = useMemo(() => [...run.events].sort((a, b) => a.frame - b.frame), [run.events]);
  useFrame(() => {
    if (!group.current || !shadow.current) return;
    const frame = playhead.current;
    const point = interpolatePosition(telemetry.ballPositions, frame, 5);
    group.current.visible = shadow.current.visible = !!point;
    if (!point) return;
    let height = 0.16;
    if (arc) {
      const end = events.findIndex(event => event.frame >= frame);
      const a = events[end - 1]; const b = events[end];
      if (a && b && b.frame > a.frame && b.frame - a.frame < telemetry.fps * 3) height += Math.sin(Math.PI * (frame - a.frame) / (b.frame - a.frame)) * 1.8;
    }
    group.current.position.set(...courtToWorld(point, height));
    shadow.current.position.set(...courtToWorld(point, 0.035));
  });
  return <>
    <group ref={group}>
      <mesh castShadow><sphereGeometry args={[0.14, 16, 12]} /><meshStandardMaterial color="#eeff7e" emissive="#c8ed4b" emissiveIntensity={0.45} roughness={0.7} /></mesh>
      <mesh><sphereGeometry args={[0.23, 12, 8]} /><meshBasicMaterial color="#d7f684" transparent opacity={0.1} depthWrite={false} /></mesh>
    </group>
    <mesh ref={shadow} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.2, 0.25, 24]} /><meshBasicMaterial color="#e8ffae" transparent opacity={0.65} /></mesh>
  </>;
}

function BounceMarkers({ run, playhead }: { run: AnalysisRun; playhead: RefObject<number> }) {
  return <>{run.events.filter(event => event.event_type === "BOUNCE" && event.court_position_m).map(event => <BouncePulse key={event.event_id} position={event.court_position_m!} frame={event.frame} fps={run.summary.fps} playhead={playhead} />)}</>;
}
function BouncePulse({ position, frame, fps, playhead }: { position: CourtPoint; frame: number; fps: number; playhead: RefObject<number> }) {
  const mesh = useRef<THREE.Mesh>(null);
  useFrame(() => { if (!mesh.current) return; const age = (playhead.current - frame) / fps; mesh.current.visible = age >= 0 && age <= 1; mesh.current.scale.setScalar(1 + Math.max(0, age) * 2); (mesh.current.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.8 * (1 - age)); });
  return <mesh ref={mesh} position={courtToWorld(position, 0.065)} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.26, 0.3, 32]} /><meshBasicMaterial color="#f5d896" transparent /></mesh>;
}

function DensityColumns({ telemetry, filter, range }: { telemetry: NormalizedTelemetry; filter: PlayerFilter; range: FrameRange }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const data = useMemo(() => {
    const p1 = densityGrid(samplePoints(telemetry, "player1", range));
    const p2 = densityGrid(samplePoints(telemetry, "player2", range));
    const peak = Math.max(1, ...p1.map(p => p.value), ...p2.map(p => p.value));
    return p1.map((p, i) => { const second = p2[i]!; const value = filter === "player1" ? p.value : filter === "player2" ? second.value : Math.max(p.value, second.value); return { ...p, intensity: value / peak, color: filter === "player2" || (filter === "combined" && second.value > p.value) ? PLAYER_COLORS[1] : PLAYER_COLORS[0] }; }).filter(p => p.intensity > 0.04);
  }, [filter, range, telemetry]);
  useEffect(() => {
    const instance = mesh.current;
    if (!instance) return;
    const object = new THREE.Object3D(); const color = new THREE.Color();
    data.forEach((p, i) => { const height = p.intensity * 3.5; object.position.set(...courtToWorld([p.x, p.y], height / 2 + 0.045)); object.scale.set(p.width * 0.7, height, p.depth * 0.7); object.updateMatrix(); instance.setMatrixAt(i, object.matrix); instance.setColorAt(i, color.set(p.color)); });
    instance.instanceMatrix.needsUpdate = true;
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
    instance.computeBoundingSphere();
  }, [data]);
  if (!data.length) return null;
  return <instancedMesh ref={mesh} args={[undefined, undefined, data.length]}><boxGeometry args={[1, 1, 1]} /><meshStandardMaterial transparent opacity={0.78} roughness={0.45} metalness={0.15} /></instancedMesh>;
}
