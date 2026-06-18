"""
streamlit_app.py — SVM Kernel Trick 3D Interactive Visualization
===============================================================
Educational Three.js visualization of how Support Vector Machines
use the kernel trick to handle nonlinearly separable data by
lifting it to higher dimensions.

Run: streamlit run streamlit_app.py

Requirements: streamlit, numpy, scikit-learn
"""

import streamlit as st
import streamlit.components.v1 as components

from src.data_gen import generate_datasets
from src.utils import to_json_compact


# ============================================================
# THREE.JS APPLICATION (embedded HTML)
# ============================================================

def build_html(data_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVM Kernel Trick Visualization</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #000;
        overflow: hidden;
        font-family: 'Courier New', Courier, monospace;
        user-select: none;
    }}
    canvas {{ display: block; }}

    #ui-overlay {{
        position: absolute;
        top: 24px;
        left: 24px;
        background: rgba(0, 0, 0, 0.75);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 24px 28px;
        color: #fff;
        max-width: 380px;
        backdrop-filter: blur(8px);
        pointer-events: none;
    }}
    #state-name {{
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 6px;
        letter-spacing: 2px;
    }}
    #formula {{
        font-size: 15px;
        color: #ffcc00;
        margin-bottom: 14px;
        word-break: break-all;
        line-height: 1.4;
    }}
    #explanation {{
        font-size: 13px;
        color: #bbb;
        line-height: 1.6;
    }}
    #explanation span.highlight {{ color: #0f0; }}
    #explanation span.warn {{ color: #ff0; }}

    #btn-bar {{
        position: absolute;
        bottom: 32px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 16px;
        z-index: 20;
    }}
    #btn-bar button {{
        padding: 14px 32px;
        font-size: 15px;
        font-weight: bold;
        letter-spacing: 1px;
        cursor: pointer;
        border: 2px solid rgba(0,255,255,0.6);
        background: rgba(0, 20, 40, 0.85);
        color: #0ff;
        border-radius: 10px;
        font-family: 'Courier New', Courier, monospace;
        transition: all 0.25s;
        backdrop-filter: blur(6px);
    }}
    #btn-bar button:hover {{
        background: rgba(0, 255, 255, 0.18);
        border-color: #0ff;
        box-shadow: 0 0 24px rgba(0,255,255,0.45);
    }}
    #btn-bar button.active {{
        background: rgba(0, 255, 255, 0.25);
        border-color: #fff;
        color: #fff;
    }}

    #note-banner {{
        position: absolute;
        bottom: 110px;
        left: 50%;
        transform: translateX(-50%);
        color: #888;
        font-size: 12px;
        font-family: 'Courier New', Courier, monospace;
        text-align: center;
        display: none;
        background: rgba(0,0,0,0.6);
        padding: 6px 18px;
        border-radius: 6px;
    }}

    #fps-counter {{
        position: absolute;
        top: 12px;
        right: 16px;
        color: #444;
        font-size: 11px;
        font-family: 'Courier New', Courier, monospace;
        z-index: 10;
    }}
</style>
</head>
<body>

<div id="ui-overlay">
    <div id="state-name">LINEAR SVM</div>
    <div id="formula">f(x) = w&#x1D40;x + b</div>
    <div id="explanation">
        Data is <span class="highlight">perfectly linearly separable</span>.<br>
        SVM finds the <span class="highlight">optimal hyperplane</span><br>
        that maximizes the margin between classes.<br><br>
        <span style="color:#0f0">&#x2501;</span> Decision Boundary: w&#x1D40;x + b = 0<br>
        <span class="warn">&#x2501;</span> Margins: w&#x1D40;x + b = &#x00B1;1<br>
        <span class="warn">&#x25EF;</span> Support Vectors (nearest points)
    </div>
</div>

<div id="btn-bar">
    <button id="btn-linear" class="active">LINEAR SVM</button>
    <button id="btn-nonlinear">NONLINEAR DATA</button>
    <button id="btn-kernel">KERNEL 3D</button>
</div>

<div id="note-banner">
    &#x26A0; This is an intuitive visualization of feature lifting and not the true infinite-dimensional RBF feature space.
</div>

<div id="fps-counter">FPS: --</div>

<script type="importmap">
{{
    "imports": {{
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';

// ============================================================
// INJECTED DATA
// ============================================================
const DATA = {data_json};
const N = DATA.n;

// ============================================================
// GLOBAL ENUMS & STATE
// ============================================================
const STATE = Object.freeze({{ LINEAR: 0, NONLINEAR: 1, KERNEL_3D: 2 }});
const STATE_NAMES = ['LINEAR', 'NONLINEAR', 'KERNEL_3D'];

let currentState = STATE.LINEAR;
let targetState = STATE.LINEAR;
let transitionProgress = 0.0;
let transitionStartMs = 0;
const T_DURATION = 2800; // ms

// Ease in-out cubic
function easeInOutCubic(t) {{
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}}

// Linear interpolation
function lerp(a, b, t) {{ return a + (b - a) * t; }}

// Smooth step (sigmoid-like)
function smoothstep(edge0, edge1, x) {{
    const t2 = Math.max(0, Math.min((x - edge0) / (edge1 - edge0), 1));
    return t2 * t2 * (3 - 2 * t2);
}}

// ============================================================
// PARTICLE DATA STRUCTURES
// ============================================================
const redParticles = [];
const blueParticles = [];

for (let i = 0; i < N; i++) {{
    const lr = DATA.linear_red[i];
    const nr = DATA.nonlinear_red[i];
    redParticles.push({{
        linearPos:    new THREE.Vector3(lr[0], lr[1], 0),
        nonlinearPos: new THREE.Vector3(nr[0], nr[1], 0),
        kernelZLin:   DATA.z_linear_red[i],
        kernelZNonlin: DATA.z_nonlinear_red[i],
        currentPos:   new THREE.Vector3(lr[0], lr[1], 0),
        isSV:         DATA.sv_indices.includes(i),
    }});
}}
for (let i = 0; i < N; i++) {{
    const lb = DATA.linear_blue[i];
    const nb = DATA.nonlinear_blue[i];
    blueParticles.push({{
        linearPos:    new THREE.Vector3(lb[0], lb[1], 0),
        nonlinearPos: new THREE.Vector3(nb[0], nb[1], 0),
        kernelZLin:   DATA.z_linear_blue[i],
        kernelZNonlin: DATA.z_nonlinear_blue[i],
        currentPos:   new THREE.Vector3(lb[0], lb[1], 0),
        isSV:         DATA.sv_indices.includes(N + i),
    }});
}}

// ============================================================
// THREE.JS CORE SETUP
// ============================================================
const W = window.innerWidth;
const H = window.innerHeight;
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
camera.position.set(0, 0, 14);
camera.lookAt(0, 0, 0);

// Ortho camera (unused for render, just for reference)
const orthoFrustum = 7;
const orthoCamera = new THREE.OrthographicCamera(
    -orthoFrustum * W / H, orthoFrustum * W / H,
    orthoFrustum, -orthoFrustum, 0.1, 100
);
orthoCamera.position.copy(camera.position);
orthoCamera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setSize(W, H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x050510);
document.body.appendChild(renderer.domElement);

// OrbitControls — disabled by default
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 4;
controls.maxDistance = 30;
controls.maxPolarAngle = Math.PI * 0.75;

// Post-processing
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

let bloomPass;
try {{
    bloomPass = new UnrealBloomPass(
        new THREE.Vector2(W, H), 0.6, 0.4, 0.75
    );
    composer.addPass(bloomPass);
}} catch (e) {{
    // Bloom not critical; degrade gracefully
    console.warn('UnrealBloomPass unavailable, skipping bloom.');
}}

// ============================================================
// LIGHTING
// ============================================================
scene.add(new THREE.AmbientLight(0x222244, 0.4));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(0, 8, 12);
scene.add(dirLight);

// ============================================================
// LAYER 1 — BACKGROUND (Starfield + Nebula)
// ============================================================
const starsGeom = new THREE.BufferGeometry();
const starCount = 1800;
const starArr = new Float32Array(starCount * 3);
for (let i = 0; i < starCount * 3; i += 3) {{
    starArr[i]     = (Math.random() - 0.5) * 45;
    starArr[i + 1] = (Math.random() - 0.5) * 45;
    starArr[i + 2] = (Math.random() - 0.5) * 22 - 11;
}}
starsGeom.setAttribute('position', new THREE.BufferAttribute(starArr, 3));
const starsMat = new THREE.PointsMaterial({{
    color: 0xccccff, size: 0.035, transparent: true, opacity: 0.75,
    blending: THREE.AdditiveBlending, depthWrite: false,
}});
const starsMesh = new THREE.Points(starsGeom, starsMat);
scene.add(starsMesh);

const nebulaGeom = new THREE.BufferGeometry();
const nebulaCount = 400;
const nebPosArr = new Float32Array(nebulaCount * 3);
const nebColArr = new Float32Array(nebulaCount * 3);
for (let i = 0; i < nebulaCount * 3; i += 3) {{
    nebPosArr[i]     = (Math.random() - 0.5) * 24;
    nebPosArr[i + 1] = (Math.random() - 0.5) * 24;
    nebPosArr[i + 2] = (Math.random() - 0.5) * 12 - 6;
    nebColArr[i]     = 0.08 + Math.random() * 0.12;
    nebColArr[i + 1] = 0.02 + Math.random() * 0.06;
    nebColArr[i + 2] = 0.15 + Math.random() * 0.25;
}}
nebulaGeom.setAttribute('position', new THREE.BufferAttribute(nebPosArr, 3));
nebulaGeom.setAttribute('color', new THREE.BufferAttribute(nebColArr, 3));
const nebulaMat = new THREE.PointsMaterial({{
    size: 0.55, vertexColors: true, transparent: true, opacity: 0.25,
    blending: THREE.AdditiveBlending, depthWrite: false,
}});
scene.add(new THREE.Points(nebulaGeom, nebulaMat));

// ============================================================
// LAYER 2 — DATA PARTICLES (BufferGeometry + PointsMaterial)
// ============================================================
const redGeom = new THREE.BufferGeometry();
const redPosArr = new Float32Array(N * 3);
redGeom.setAttribute('position', new THREE.BufferAttribute(redPosArr, 3));
const redMat = new THREE.PointsMaterial({{
    color: 0xff2233, size: 0.20, blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true,
}});
const redPoints = new THREE.Points(redGeom, redMat);
scene.add(redPoints);

const blueGeom = new THREE.BufferGeometry();
const bluePosArr = new Float32Array(N * 3);
blueGeom.setAttribute('position', new THREE.BufferAttribute(bluePosArr, 3));
const blueMat = new THREE.PointsMaterial({{
    color: 0x2266ff, size: 0.20, blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true,
}});
const bluePoints = new THREE.Points(blueGeom, blueMat);
scene.add(bluePoints);

// ============================================================
// LAYER 3 — DECISION BOUNDARY & MARGIN LINES
// ============================================================
const dbGroup = new THREE.Group();
scene.add(dbGroup);

function lineMesh(pA, pB, color, width, opacity) {{
    const dir = new THREE.Vector3().subVectors(pB, pA);
    const len = dir.length();
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    const geom = new THREE.CylinderGeometry(width, width, len, 6, 1);
    const mat = new THREE.MeshBasicMaterial({{
        color, transparent: true, opacity, depthTest: true,
    }});
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.copy(mid);
    // Align cylinder (Y-up) to dir
    const quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0), dir.normalize()
    );
    mesh.setRotationFromQuaternion(quat);
    return mesh;
}}

function buildDecisionBoundary() {{
    while (dbGroup.children.length > 0) dbGroup.remove(dbGroup.children[0]);

    const p1  = new THREE.Vector3(DATA.db_line[0][0], DATA.db_line[0][1], 0.02);
    const p2  = new THREE.Vector3(DATA.db_line[1][0], DATA.db_line[1][1], 0.02);
    const mp1 = new THREE.Vector3(DATA.margin_pos[0][0], DATA.margin_pos[0][1], 0.01);
    const mp2 = new THREE.Vector3(DATA.margin_pos[1][0], DATA.margin_pos[1][1], 0.01);
    const mn1 = new THREE.Vector3(DATA.margin_neg[0][0], DATA.margin_neg[0][1], 0.01);
    const mn2 = new THREE.Vector3(DATA.margin_neg[1][0], DATA.margin_neg[1][1], 0.01);

    dbGroup.add(lineMesh(p1, p2,   0x00ff88, 0.08, 1.0));  // decision boundary
    dbGroup.add(lineMesh(mp1, mp2, 0xffcc00, 0.04, 0.9));  // margin +
    dbGroup.add(lineMesh(mn1, mn2, 0xffcc00, 0.04, 0.9));  // margin -
}}
buildDecisionBoundary();

// ============================================================
// SUPPORT VECTOR INDICATORS (glowing rings)
// ============================================================
const svGroup = new THREE.Group();
scene.add(svGroup);
const svRingGeom = new THREE.TorusGeometry(0.32, 0.05, 8, 20);

function refreshSVIndicators(opacity) {{
    while (svGroup.children.length > 0) svGroup.remove(svGroup.children[0]);
    const allParts = redParticles.concat(blueParticles);
    for (const p of allParts) {{
        if (!p.isSV) continue;
        const mat = new THREE.MeshBasicMaterial({{
            color: 0xffff44, transparent: true, opacity: opacity,
            depthTest: true,
        }});
        const ring = new THREE.Mesh(svRingGeom, mat);
        ring.position.copy(p.currentPos);
        ring.position.z += 0.03;
        svGroup.add(ring);
    }}
}}
refreshSVIndicators(0.85);

// ============================================================
// LAYER 4 — KERNEL TRANSFORMATION GRID (warping surface)
// ============================================================
const kernelGridGroup = new THREE.Group();
kernelGridGroup.visible = false;
scene.add(kernelGridGroup);

const GRID_RES = 28;
const GRID_EXT = 7.5;
const gridNodes = [];
const gridDotGeom = new THREE.SphereGeometry(0.045, 4, 4);
const gridDotMat = new THREE.MeshBasicMaterial({{
    color: 0x8888ff, transparent: true, opacity: 0.55, depthTest: true,
}});

for (let i = 0; i <= GRID_RES; i++) {{
    for (let j = 0; j <= GRID_RES; j++) {{
        const fx = (i / GRID_RES - 0.5) * 2 * GRID_EXT;
        const fy = (j / GRID_RES - 0.5) * 2 * GRID_EXT;
        const m = new THREE.Mesh(gridDotGeom, gridDotMat.clone());
        m.position.set(fx, fy, 0);
        m.userData = {{ baseX: fx, baseY: fy }};
        kernelGridGroup.add(m);
        gridNodes.push(m);
    }}
}}

// ============================================================
// LAYER 5 — HYPERPLANE (separating plane in kernel space)
// ============================================================
const hyperplaneGroup = new THREE.Group();
hyperplaneGroup.visible = false;
scene.add(hyperplaneGroup);

const hpGeom = new THREE.PlaneGeometry(10, 10);
const hpMat = new THREE.MeshBasicMaterial({{
    color: 0x00ffff, side: THREE.DoubleSide,
    transparent: true, opacity: 0.28, depthWrite: false,
}});
const hpPlane = new THREE.Mesh(hpGeom, hpMat);
hpPlane.position.z = DATA.hyperplane_z;
hyperplaneGroup.add(hpPlane);

const hpRingGeom = new THREE.TorusGeometry(4.2, 0.06, 12, 80);
const hpRingMat = new THREE.MeshBasicMaterial({{
    color: 0x00ffff, transparent: true, opacity: 0.5, depthTest: true,
}});
const hpRingMesh = new THREE.Mesh(hpRingGeom, hpRingMat);
hpRingMesh.position.z = DATA.hyperplane_z;
hyperplaneGroup.add(hpRingMesh);

// ============================================================
// LAYER 6 — PROJECTION (circular boundary + floor projection)
// ============================================================
const projectionGroup = new THREE.Group();
projectionGroup.visible = false;
scene.add(projectionGroup);

const projCircleGeom = new THREE.TorusGeometry(2.0, 0.05, 16, 80);
const projCircleMat = new THREE.MeshBasicMaterial({{
    color: 0x00ff88, transparent: true, opacity: 0.75, depthTest: true,
}});
const projCircle = new THREE.Mesh(projCircleGeom, projCircleMat);
projectionGroup.add(projCircle);

const projLinesGroup = new THREE.Group();
projectionGroup.add(projLinesGroup);

// ============================================================
// AXIS HELPER (faint)
// ============================================================
const axisGroup = new THREE.Group();
const axisLen = 8;
function thinLine(start, end, color) {{
    return lineMesh(start, end, color, 0.02, 0.25);
}}
axisGroup.add(thinLine(new THREE.Vector3(-axisLen,0,0), new THREE.Vector3(axisLen,0,0), 0x444444));
axisGroup.add(thinLine(new THREE.Vector3(0,-axisLen,0), new THREE.Vector3(0,axisLen,0), 0x444444));
axisGroup.add(thinLine(new THREE.Vector3(0,0,-3), new THREE.Vector3(0,0,3), 0x444444));
scene.add(axisGroup);

// ============================================================
// POSITION UPDATE HELPERS
// ============================================================

function setParticlePositions(morphT) {{
    // morphT: 0 = linear positions, 1 = nonlinear positions (2D only)
    const rArr = redGeom.attributes.position.array;
    const bArr = blueGeom.attributes.position.array;

    for (let i = 0; i < N; i++) {{
        const r = redParticles[i];
        const b = blueParticles[i];

        const rx = lerp(r.linearPos.x, r.nonlinearPos.x, morphT);
        const ry = lerp(r.linearPos.y, r.nonlinearPos.y, morphT);
        const bx = lerp(b.linearPos.x, b.nonlinearPos.x, morphT);
        const by = lerp(b.linearPos.y, b.nonlinearPos.y, morphT);

        r.currentPos.set(rx, ry, r.currentPos.z);
        b.currentPos.set(bx, by, b.currentPos.z);

        const ri = i * 3;
        const bi = i * 3;
        rArr[ri]     = rx; rArr[ri + 1] = ry; rArr[ri + 2] = r.currentPos.z;
        bArr[bi]     = bx; bArr[bi + 1] = by; bArr[bi + 2] = b.currentPos.z;
    }}
    redGeom.attributes.position.needsUpdate = true;
    blueGeom.attributes.position.needsUpdate = true;
}}

function applyKernelLift(kernelT) {{
    // kernelT: 0 = flat (z=0), 1 = fully lifted
    const rArr = redGeom.attributes.position.array;
    const bArr = blueGeom.attributes.position.array;

    for (let i = 0; i < N; i++) {{
        const r = redParticles[i];
        const b = blueParticles[i];
        // Use nonlinear kernel Z for the current morph state
        const rz = lerp(0, r.kernelZNonlin, kernelT);
        const bz = lerp(0, b.kernelZNonlin, kernelT);
        r.currentPos.z = rz;
        b.currentPos.z = bz;
        rArr[i * 3 + 2] = rz;
        bArr[i * 3 + 2] = bz;
    }}
    redGeom.attributes.position.needsUpdate = true;
    blueGeom.attributes.position.needsUpdate = true;
}}

function updateGridLift(kernelT) {{
    for (const node of gridNodes) {{
        const x = node.userData.baseX;
        const y = node.userData.baseY;
        const z = Math.exp(-(x * x + y * y)) * kernelT;
        node.position.z = z;
        // Adjust opacity based on height
        node.material.opacity = 0.2 + kernelT * 0.35;
    }}
}}

function updateCamera3D(rotT) {{
    // rotT: 0 = top-down, 1 = angled 3D
    const tx = rotT * 5.5;
    const ty = rotT * 5.5;
    const tz = 14 - rotT * 5;
    camera.position.set(tx, ty, tz);
    const lookZ = lerp(0, DATA.hyperplane_z, rotT);
    camera.lookAt(0, 0, lookZ);
}}

function updateProjectionLines(opacity) {{
    while (projLinesGroup.children.length > 0) projLinesGroup.remove(projLinesGroup[0]);
    if (opacity <= 0) return;

    // Sample every 4th particle for perf
    const allParts = redParticles.concat(blueParticles);
    for (let i = 0; i < allParts.length; i += 4) {{
        const p = allParts[i];
        const top = p.currentPos.clone();
        const bot = p.currentPos.clone();
        bot.z = 0;
        const m = lineMesh(top, bot, p === redParticles[i] || (i >= N && p.isSV) ? 0xff3344 : 0x3355ff, 0.015, opacity * 0.35);
        projLinesGroup.add(m);
    }}
}}

// ============================================================
// DB / SV / PROJ visibility helpers
// ============================================================

function setDBOpacity(o) {{
    dbGroup.children.forEach(c => {{ if (c.material) c.material.opacity = o * c.material.opacity / Math.max(c.material.opacity, 0.01); }});
    // Fix: set all to same opacity
    for (const c of dbGroup.children) {{
        if (c.material) c.material.opacity = o * (c.material.color.getHex() === 0x00ff88 ? 1.0 : 0.9);
    }}
}}

function setLayerVisibility(group, visible) {{
    group.visible = visible;
    if (!visible) group.children.forEach(c => {{ if (c.material) c.material.opacity = 0; }});
}}

// ============================================================
// UI UPDATE
// ============================================================

function updateUIOverlay() {{
    const elName  = document.getElementById('state-name');
    const elForm  = document.getElementById('formula');
    const elExpl  = document.getElementById('explanation');
    const note    = document.getElementById('note-banner');

    switch (currentState) {{
        case STATE.LINEAR:
            elName.textContent = 'LINEAR SVM';
            elName.style.color = '#0ff';
            elForm.innerHTML = 'f(x) = <b>w</b><sup>T</sup>x + b';
            elExpl.innerHTML = `
                Data is <span class="highlight">perfectly linearly separable</span>.<br>
                SVM finds the <span class="highlight">optimal hyperplane</span><br>
                that maximizes the margin between classes.<br><br>
                <span style="color:#0f0">&#x2501;</span> Decision Boundary: w<sup>T</sup>x + b = 0<br>
                <span class="warn">&#x2501;</span> Margins: w<sup>T</sup>x + b = &plusmn;1<br>
                <span class="warn">&#x25EF;</span> Support Vectors (nearest points)
            `;
            note.style.display = 'none';
            break;
        case STATE.NONLINEAR:
            elName.textContent = 'NONLINEAR DATA';
            elName.style.color = '#f0f';
            elForm.innerHTML = 'No linear separator exists in &#x211D;<sup>2</sup>';
            elExpl.innerHTML = `
                Data is <span class="warn">NOT</span> linearly separable.<br>
                Red cluster at center, Blue ring around it.<br>
                No straight line can separate them.<br><br>
                <span class="warn">&#x2753;</span> How can SVM handle this?<br>
                <span style="color:#888;">&#x2192;</span> The <b>Kernel Trick</b> lifts data to a<br>
                higher-dimensional space where it<br>
                <span class="highlight">becomes</span> linearly separable!
            `;
            note.style.display = 'none';
            break;
        case STATE.KERNEL_3D:
            elName.textContent = 'KERNEL TRICK (3D)';
            elName.style.color = '#ff0';
            elForm.innerHTML = '&#x3A6;(x<sub>1</sub>,x<sub>2</sub>) = (x<sub>1</sub>, x<sub>2</sub>, e<sup>&#x2212;(x<sub>1</sub><sup>2</sup>+x<sub>2</sub><sup>2</sup>)</sup>)';
            elExpl.innerHTML = `
                Data <span class="highlight">lifted to 3D</span> via kernel!<br>
                K(x,y) = exp(&#x2212;&#x3B3;||x&#x2212;y||<sup>2</sup>)<br><br>
                <span style="color:#0ff">&#x25A1;</span> Separating hyperplane in 3D<br>
                <span style="color:#0f0">&#x25CB;</span> Circular boundary projected back<br><br>
                <span class="highlight">In higher dimensions, data<br>becomes linearly separable!</span>
            `;
            note.style.display = 'block';
            break;
    }}

    // Highlight active button
    ['btn-linear','btn-nonlinear','btn-kernel'].forEach(id => {{
        document.getElementById(id).classList.remove('active');
    }});
    const activeMap = {{ 0: 'btn-linear', 1: 'btn-nonlinear', 2: 'btn-kernel' }};
    document.getElementById(activeMap[currentState]).classList.add('active');
}}

function updateCameraForState() {{
    if (currentState === STATE.KERNEL_3D) {{
        updateCamera3D(1);
        controls.enabled = true;
    }} else {{
        camera.position.set(0, 0, 14);
        camera.lookAt(0, 0, 0);
        controls.enabled = false;
        controls.target.set(0, 0, 0);
        controls.update();
    }}
}}

// ============================================================
// TRANSITION ENGINE
// ============================================================

function requestTransition(toState) {{
    if (targetState === toState && transitionProgress < 1 && transitionProgress > 0) return; // already going there
    if (currentState === toState) return;

    targetState = toState;
    transitionProgress = 0;
    transitionStartMs = performance.now();
}}

function tickTransition(now) {{
    if (targetState === currentState) return;

    const elapsed = now - transitionStartMs;
    transitionProgress = Math.min(elapsed / T_DURATION, 1.0);
    const t = easeInOutCubic(transitionProgress);

    const from = currentState;

    // ----- LINEAR -> NONLINEAR -----
    if (from === STATE.LINEAR && targetState === STATE.NONLINEAR) {{
        setParticlePositions(t);
        setDBOpacity(1 - t);
        refreshSVIndicators(0.85 * (1 - t));
    }}

    // ----- NONLINEAR -> KERNEL_3D -----
    if (from === STATE.NONLINEAR && targetState === STATE.KERNEL_3D) {{
        // Phase 1 (0.0-0.35): kernel lift
        const liftT = smoothstep(0.0, 0.35, t);
        applyKernelLift(liftT);

        // Phase 2 (0.2-1.0): camera + layers
        const showT = smoothstep(0.2, 1.0, t);

        kernelGridGroup.visible = showT > 0.05;
        updateGridLift(Math.min(showT * 1.4, 1.0));

        hyperplaneGroup.visible = showT > 0.25;
        hpMat.opacity = 0.28 * smoothstep(0.25, 0.5, t);
        hpRingMat.opacity = 0.5 * smoothstep(0.25, 0.5, t);

        projectionGroup.visible = showT > 0.35;
        projCircleMat.opacity = 0.75 * smoothstep(0.35, 0.6, t);
        projCircle.visible = projCircleMat.opacity > 0.01;
        updateProjectionLines(smoothstep(0.35, 0.7, t));

        updateCamera3D(showT);
        controls.enabled = showT > 0.5;

        document.getElementById('note-banner').style.display = showT > 0.4 ? 'block' : 'none';
    }}

    // ----- LINEAR -> KERNEL_3D (chained: LINEAR->NONLINEAR->KERNEL_3D internally) -----
    if (from === STATE.LINEAR && targetState === STATE.KERNEL_3D) {{
        // Two-phase transition within same animation
        const phase1End = 0.35;
        const phase2Start = 0.25; // overlap slightly

        if (t <= phase1End) {{
            // Phase 1: morph to nonlinear
            const pt = Math.min(t / phase1End, 1.0);
            setParticlePositions(pt);
            setDBOpacity(1 - pt);
            refreshSVIndicators(0.85 * (1 - pt));
        }} else {{
            setParticlePositions(1);
            setDBOpacity(0);
            refreshSVIndicators(0);
        }}

        // Phase 2: kernel lift + camera
        const liftT = smoothstep(phase2Start, 0.65, t);
        applyKernelLift(liftT);
        kernelGridGroup.visible = liftT > 0.05;
        updateGridLift(Math.min(liftT * 1.4, 1.0));

        const showT = smoothstep(phase2Start + 0.05, 1.0, t);
        hyperplaneGroup.visible = showT > 0.2;
        hpMat.opacity = 0.28 * smoothstep(0.2, 0.45, showT);
        hpRingMat.opacity = 0.5 * smoothstep(0.2, 0.45, showT);

        projectionGroup.visible = showT > 0.3;
        projCircleMat.opacity = 0.75 * smoothstep(0.3, 0.55, showT);
        projCircle.visible = projCircleMat.opacity > 0.01;
        updateProjectionLines(smoothstep(0.3, 0.65, showT));

        updateCamera3D(showT);
        controls.enabled = showT > 0.45;
        document.getElementById('note-banner').style.display = showT > 0.35 ? 'block' : 'none';
    }}

    // Completion
    if (transitionProgress >= 1.0) {{
        currentState = targetState;
        transitionProgress = 0;
        updateCameraForState();
        updateUIOverlay();
    }}
}}

// ============================================================
// RESET TO LINEAR (instant)
// ============================================================
function resetToLinear() {{
    currentState = STATE.LINEAR;
    targetState = STATE.LINEAR;
    transitionProgress = 0;
    setParticlePositions(0);
    applyKernelLift(0);
    setDBOpacity(1);
    refreshSVIndicators(0.85);
    setLayerVisibility(kernelGridGroup, false);
    setLayerVisibility(hyperplaneGroup, false);
    setLayerVisibility(projectionGroup, false);
    updateCameraForState();
    updateProjectionLines(0);
    document.getElementById('note-banner').style.display = 'none';
    updateUIOverlay();
}}

// ============================================================
// BUTTON HANDLERS
// ============================================================
document.getElementById('btn-linear').addEventListener('click', () => {{
    resetToLinear();
}});

document.getElementById('btn-nonlinear').addEventListener('click', () => {{
    if (currentState === STATE.NONLINEAR) return;
    if (currentState === STATE.KERNEL_3D) {{
        // Instant reset to linear, then animate to nonlinear
        resetToLinear();
        setTimeout(() => requestTransition(STATE.NONLINEAR), 80);
    }} else {{
        requestTransition(STATE.NONLINEAR);
    }}
}});

document.getElementById('btn-kernel').addEventListener('click', () => {{
    if (currentState === STATE.KERNEL_3D) return;
    if (currentState === STATE.LINEAR) {{
        requestTransition(STATE.KERNEL_3D);
    }} else {{
        requestTransition(STATE.KERNEL_3D);
    }}
}});

// ============================================================
// FPS COUNTER
// ============================================================
let frameCount = 0;
let lastFpsTime = performance.now();
const fpsEl = document.getElementById('fps-counter');

function updateFPS(now) {{
    frameCount++;
    if (now - lastFpsTime >= 1000) {{
        const fps = Math.round(frameCount / ((now - lastFpsTime) / 1000));
        fpsEl.textContent = 'FPS: ' + fps;
        frameCount = 0;
        lastFpsTime = now;
    }}
}}

// ============================================================
// MAIN ANIMATION LOOP
// ============================================================
function animate(timestamp) {{
    requestAnimationFrame(animate);

    tickTransition(timestamp);

    starsMesh.rotation.y += 0.00015;
    starsMesh.rotation.x += 0.00008;

    if (currentState === STATE.KERNEL_3D) {{
        hpRingMesh.rotation.z += 0.003;
        projCircle.rotation.z += 0.002;
    }}

    refreshSVIndicators(
        currentState === STATE.LINEAR ? 0.85 :
        currentState === STATE.NONLINEAR ? 0 :
        0
    );

    controls.update();

    if (bloomPass) {{
        composer.render();
    }} else {{
        renderer.render(scene, camera);
    }}

    updateFPS(timestamp);
}}

// ============================================================
// RESIZE HANDLER
// ============================================================
window.addEventListener('resize', () => {{
    const w2 = window.innerWidth;
    const h2 = window.innerHeight;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
    if (bloomPass) composer.setSize(w2, h2);
}});

// ============================================================
// INIT
// ============================================================
setParticlePositions(0);
updateUIOverlay();
updateCameraForState();
requestAnimationFrame(animate);
</script>
</body>
</html>"""

# ============================================================
# STREAMLIT ENTRY POINT
# ============================================================

def main():
    st.set_page_config(
        page_title="SVM Kernel Trick – 3D Visualization",
        page_icon="🔮",
        layout="wide",
    )
    st.title("🔮 SVM Kernel Trick: 3D Interactive Visualization")
    st.markdown(
        """
        **Educational visualization** of how Support Vector Machines use the
        **kernel trick** to handle nonlinearly separable data by lifting it
        into a higher-dimensional space where a linear separator exists.

        Use the buttons inside the 3D view to step through the states:
        **LINEAR SVM** → **NONLINEAR DATA** → **KERNEL 3D**.
        In the 3D state you can drag to rotate, scroll to zoom, and right-drag to pan.
        """
    )

    data = generate_datasets()
    data_json = to_json_compact(data)
    html = build_html(data_json)
    components.html(html, height=720, scrolling=False)


if __name__ == "__main__":
    main()
