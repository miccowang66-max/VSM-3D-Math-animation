"""
streamlit_app.py — SVM 核方法 3D 互動視覺化
============================================
展示支持向量機 (SVM) 如何透過核方法 (Kernel Trick)
將非線性可分的資料映射到高維空間中，使其變得線性可分。

執行: streamlit run streamlit_app.py
依賴: streamlit, numpy, scikit-learn
"""

import streamlit as st
import streamlit.components.v1 as components

from src.data_gen import generate_datasets
from src.utils import to_json_compact


# ============================================================
# THREE.JS 嵌入式應用 (HTML/JS)
# ============================================================

def build_html(data_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVM 核方法 3D 視覺化</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #000;
        overflow: hidden;
        font-family: 'Microsoft YaHei', 'PingFang TC', 'Noto Sans TC', sans-serif;
        user-select: none;
    }}
    canvas {{ display: block; }}

    #loading {{
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        color: #0ff; font-size: 20px; z-index: 100;
        text-align: center;
    }}
    #loading .spinner {{
        width: 48px; height: 48px;
        border: 3px solid rgba(0,255,255,0.2);
        border-top-color: #0ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 16px auto;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    #error-msg {{
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        color: #f44; font-size: 16px; z-index: 100;
        text-align: center; display: none;
        background: rgba(0,0,0,0.8); padding: 20px 30px;
        border-radius: 10px; border: 1px solid #f44;
    }}

    #ui-overlay {{
        position: absolute; top: 24px; left: 24px;
        background: rgba(0,0,0,0.75);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px; padding: 24px 28px;
        color: #fff; max-width: 420px;
        backdrop-filter: blur(8px); pointer-events: none;
        display: none;
    }}
    #state-name {{
        font-size: 28px; font-weight: bold; margin-bottom: 6px; letter-spacing: 2px;
    }}
    #formula {{
        font-size: 15px; color: #ffcc00; margin-bottom: 14px;
        word-break: break-all; line-height: 1.4;
    }}
    #explanation {{
        font-size: 14px; color: #bbb; line-height: 1.7;
    }}
    #explanation span.hl {{ color: #0f0; }}
    #explanation span.wn {{ color: #ff0; }}

    #btn-bar {{
        position: absolute; bottom: 32px; left: 50%;
        transform: translateX(-50%);
        display: flex; gap: 16px; z-index: 20;
        display: none;
    }}
    #btn-bar button {{
        padding: 14px 28px; font-size: 15px; font-weight: bold;
        letter-spacing: 1px; cursor: pointer;
        border: 2px solid rgba(0,255,255,0.6);
        background: rgba(0,20,40,0.85); color: #0ff;
        border-radius: 10px;
        font-family: 'Microsoft YaHei', 'PingFang TC', 'Noto Sans TC', sans-serif;
        transition: all 0.25s; backdrop-filter: blur(6px);
        white-space: nowrap;
    }}
    #btn-bar button:hover {{
        background: rgba(0,255,255,0.18); border-color: #0ff;
        box-shadow: 0 0 24px rgba(0,255,255,0.45);
    }}
    #btn-bar button.active {{
        background: rgba(0,255,255,0.25); border-color: #fff; color: #fff;
    }}

    #note-banner {{
        position: absolute; bottom: 110px; left: 50%;
        transform: translateX(-50%);
        color: #888; font-size: 13px;
        font-family: 'Microsoft YaHei', 'PingFang TC', 'Noto Sans TC', sans-serif;
        text-align: center; display: none;
        background: rgba(0,0,0,0.6); padding: 8px 20px;
        border-radius: 6px; white-space: nowrap;
    }}

    #fps-counter {{
        position: absolute; top: 12px; right: 16px;
        color: #444; font-size: 11px;
        font-family: monospace; z-index: 10;
        display: none;
    }}
</style>
</head>
<body>

<div id="loading">
    <div class="spinner"></div>
    <div>正在載入 3D 場景...</div>
</div>
<div id="error-msg"></div>

<div id="ui-overlay">
    <div id="state-name">線性 SVM</div>
    <div id="formula">f(x) = wᵀx + b</div>
    <div id="explanation">
        資料是 <span class="hl">完美線性可分</span> 的。<br>
        SVM 找到 <span class="hl">最佳超平面</span>，<br>
        最大化兩個類別之間的邊界距離。<br><br>
        <span style="color:#0f0">▬</span> 決策邊界：wᵀx + b = 0<br>
        <span class="wn">▬</span> 邊界線：wᵀx + b = ±1<br>
        <span class="wn">◯</span> 支持向量（最近的點）
    </div>
</div>

<div id="btn-bar">
    <button id="btn-linear" class="active">線性 SVM</button>
    <button id="btn-nonlinear">非線性資料</button>
    <button id="btn-kernel">核方法 3D</button>
</div>

<div id="note-banner">
    ⚠ 此為特徵空間提升的直觀示意，並非真實的無限維 RBF 特徵空間
</div>

<div id="fps-counter">FPS: --</div>

<!-- Three.js r152.2 from jsdelivr (global, no importmap needed) -->
<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/controls/OrbitControls.js">
</script>

<script>
(function() {{
'use strict';

// ============================================================
// 顯示錯誤的輔助函式
// ============================================================
function showError(msg) {{
    var el = document.getElementById('error-msg');
    el.style.display = 'block';
    el.textContent = '⚠ 載入失敗：' + msg;
    document.getElementById('loading').style.display = 'none';
}}

function hideLoading() {{
    document.getElementById('loading').style.display = 'none';
}}

function showUI() {{
    document.getElementById('ui-overlay').style.display = 'block';
    document.getElementById('btn-bar').style.display = 'flex';
    document.getElementById('fps-counter').style.display = 'block';
}}

// ============================================================
// 檢查 Three.js 是否載入成功
// ============================================================
if (typeof THREE === 'undefined') {{
    showError('無法載入 Three.js 函式庫，請檢查網路連線');
    throw new Error('THREE not loaded');
}}

// ============================================================
// 注入資料
// ============================================================
var DATA = {data_json};
var N = DATA.n;

// ============================================================
// 全域狀態
// ============================================================
var STATE = {{ LINEAR: 0, NONLINEAR: 1, KERNEL_3D: 2 }};
var currentState = STATE.LINEAR;
var targetState = STATE.LINEAR;
var transitionProgress = 0.0;
var transitionStartMs = 0;
var T_DURATION = 2800;

function easeInOutCubic(t) {{
    return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3)/2;
}}
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function smoothstep(e0, e1, x) {{
    var t2 = Math.max(0, Math.min((x - e0)/(e1 - e0), 1));
    return t2 * t2 * (3 - 2*t2);
}}

// ============================================================
// 粒子資料結構
// ============================================================
var redParticles = [];
var blueParticles = [];

for (var i = 0; i < N; i++) {{
    var lr = DATA.linear_red[i];
    var nr = DATA.nonlinear_red[i];
    redParticles.push({{
        linearPos:    new THREE.Vector3(lr[0], lr[1], 0),
        nonlinearPos: new THREE.Vector3(nr[0], nr[1], 0),
        kernelZNonlin: DATA.z_nonlinear_red[i],
        currentPos:   new THREE.Vector3(lr[0], lr[1], 0),
        isSV:         DATA.sv_indices.indexOf(i) !== -1,
    }});
}}
for (var i = 0; i < N; i++) {{
    var lb = DATA.linear_blue[i];
    var nb = DATA.nonlinear_blue[i];
    blueParticles.push({{
        linearPos:    new THREE.Vector3(lb[0], lb[1], 0),
        nonlinearPos: new THREE.Vector3(nb[0], nb[1], 0),
        kernelZNonlin: DATA.z_nonlinear_blue[i],
        currentPos:   new THREE.Vector3(lb[0], lb[1], 0),
        isSV:         DATA.sv_indices.indexOf(N + i) !== -1,
    }});
}}

// ============================================================
// THREE.JS 場景初始化
// ============================================================
var W = window.innerWidth;
var H = window.innerHeight;
var scene = new THREE.Scene();

var camera = new THREE.PerspectiveCamera(50, W/H, 0.1, 100);
camera.position.set(0, 0, 14);
camera.lookAt(0, 0, 0);

var renderer;
try {{
    renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x050510);
    document.body.appendChild(renderer.domElement);
}} catch(e) {{
    showError('WebGL 不受支援，請使用較新的瀏覽器');
    throw e;
}}

// OrbitControls (預設關閉)
var controls;
try {{
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enabled = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4;
    controls.maxDistance = 30;
    controls.maxPolarAngle = Math.PI * 0.75;
}} catch(e) {{
    showError('OrbitControls 載入失敗');
    throw e;
}}

// ============================================================
// 燈光
// ============================================================
scene.add(new THREE.AmbientLight(0x222244, 0.4));
var dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(0, 8, 12);
scene.add(dirLight);

// ============================================================
// 圖層 1 — 背景星空
// ============================================================
var starsGeom = new THREE.BufferGeometry();
var starCount = 1800;
var starArr = new Float32Array(starCount * 3);
for (var si = 0; si < starCount * 3; si += 3) {{
    starArr[si]     = (Math.random() - 0.5) * 45;
    starArr[si + 1] = (Math.random() - 0.5) * 45;
    starArr[si + 2] = (Math.random() - 0.5) * 22 - 11;
}}
starsGeom.setAttribute('position', new THREE.BufferAttribute(starArr, 3));
var starsMesh = new THREE.Points(starsGeom, new THREE.PointsMaterial({{
    color: 0xccccff, size: 0.035, transparent: true, opacity: 0.75,
    blending: THREE.AdditiveBlending, depthWrite: false,
}}));
scene.add(starsMesh);

// 星雲
var nebulaGeom = new THREE.BufferGeometry();
var nebCount = 400;
var nebPos = new Float32Array(nebCount * 3);
var nebCol = new Float32Array(nebCount * 3);
for (var ni = 0; ni < nebCount * 3; ni += 3) {{
    nebPos[ni]     = (Math.random() - 0.5) * 24;
    nebPos[ni + 1] = (Math.random() - 0.5) * 24;
    nebPos[ni + 2] = (Math.random() - 0.5) * 12 - 6;
    nebCol[ni]     = 0.08 + Math.random() * 0.12;
    nebCol[ni + 1] = 0.02 + Math.random() * 0.06;
    nebCol[ni + 2] = 0.15 + Math.random() * 0.25;
}}
nebulaGeom.setAttribute('position', new THREE.BufferAttribute(nebPos, 3));
nebulaGeom.setAttribute('color', new THREE.BufferAttribute(nebCol, 3));
scene.add(new THREE.Points(nebulaGeom, new THREE.PointsMaterial({{
    size: 0.55, vertexColors: true, transparent: true, opacity: 0.25,
    blending: THREE.AdditiveBlending, depthWrite: false,
}})));

// ============================================================
// 圖層 2 — 資料粒子
// ============================================================
var redGeom = new THREE.BufferGeometry();
var redPosArr = new Float32Array(N * 3);
redGeom.setAttribute('position', new THREE.BufferAttribute(redPosArr, 3));
var redPoints = new THREE.Points(redGeom, new THREE.PointsMaterial({{
    color: 0xff2233, size: 0.20, blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true,
}}));
scene.add(redPoints);

var blueGeom = new THREE.BufferGeometry();
var bluePosArr = new Float32Array(N * 3);
blueGeom.setAttribute('position', new THREE.BufferAttribute(bluePosArr, 3));
var bluePoints = new THREE.Points(blueGeom, new THREE.PointsMaterial({{
    color: 0x2266ff, size: 0.20, blending: THREE.AdditiveBlending,
    depthWrite: false, transparent: true,
}}));
scene.add(bluePoints);

// ============================================================
// 圖層 3 — 決策邊界線
// ============================================================
var dbGroup = new THREE.Group();
scene.add(dbGroup);

function lineMesh(pA, pB, color, width, opacity) {{
    var dir = new THREE.Vector3().subVectors(pB, pA);
    var len = dir.length();
    var mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    var geom = new THREE.CylinderGeometry(width, width, len, 6, 1);
    var mat = new THREE.MeshBasicMaterial({{
        color: color, transparent: true, opacity: opacity, depthTest: true,
    }});
    var mesh = new THREE.Mesh(geom, mat);
    mesh.position.copy(mid);
    var quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0), dir.normalize()
    );
    mesh.setRotationFromQuaternion(quat);
    return mesh;
}}

function buildDecisionBoundary() {{
    while (dbGroup.children.length > 0) dbGroup.remove(dbGroup.children[0]);
    var p1  = new THREE.Vector3(DATA.db_line[0][0], DATA.db_line[0][1], 0.02);
    var p2  = new THREE.Vector3(DATA.db_line[1][0], DATA.db_line[1][1], 0.02);
    var mp1 = new THREE.Vector3(DATA.margin_pos[0][0], DATA.margin_pos[0][1], 0.01);
    var mp2 = new THREE.Vector3(DATA.margin_pos[1][0], DATA.margin_pos[1][1], 0.01);
    var mn1 = new THREE.Vector3(DATA.margin_neg[0][0], DATA.margin_neg[0][1], 0.01);
    var mn2 = new THREE.Vector3(DATA.margin_neg[1][0], DATA.margin_neg[1][1], 0.01);
    dbGroup.add(lineMesh(p1, p2,   0x00ff88, 0.08, 1.0));
    dbGroup.add(lineMesh(mp1, mp2, 0xffcc00, 0.04, 0.9));
    dbGroup.add(lineMesh(mn1, mn2, 0xffcc00, 0.04, 0.9));
}}
buildDecisionBoundary();

// ============================================================
// 支持向量指示器
// ============================================================
var svGroup = new THREE.Group();
scene.add(svGroup);
var svRingGeom = new THREE.TorusGeometry(0.32, 0.05, 8, 20);

function refreshSVIndicators(opacity) {{
    while (svGroup.children.length > 0) svGroup.remove(svGroup.children[0]);
    var allParts = redParticles.concat(blueParticles);
    for (var pj = 0; pj < allParts.length; pj++) {{
        if (!allParts[pj].isSV) continue;
        var rm = new THREE.MeshBasicMaterial({{
            color: 0xffff44, transparent: true, opacity: opacity, depthTest: true,
        }});
        var ring = new THREE.Mesh(svRingGeom, rm);
        ring.position.copy(allParts[pj].currentPos);
        ring.position.z += 0.03;
        svGroup.add(ring);
    }}
}}
refreshSVIndicators(0.85);

// ============================================================
// 圖層 4 — 核轉換網格
// ============================================================
var kernelGridGroup = new THREE.Group();
kernelGridGroup.visible = false;
scene.add(kernelGridGroup);

var GRID_RES = 28;
var GRID_EXT = 7.5;
var gridNodes = [];
var gridDotGeom = new THREE.SphereGeometry(0.045, 4, 4);

for (var gi = 0; gi <= GRID_RES; gi++) {{
    for (var gj = 0; gj <= GRID_RES; gj++) {{
        var fx = (gi/GRID_RES - 0.5) * 2 * GRID_EXT;
        var fy = (gj/GRID_RES - 0.5) * 2 * GRID_EXT;
        var gm = new THREE.Mesh(gridDotGeom, new THREE.MeshBasicMaterial({{
            color: 0x8888ff, transparent: true, opacity: 0.55, depthTest: true,
        }}));
        gm.position.set(fx, fy, 0);
        gm.userData = {{ baseX: fx, baseY: fy }};
        kernelGridGroup.add(gm);
        gridNodes.push(gm);
    }}
}}

// ============================================================
// 圖層 5 — 超平面
// ============================================================
var hyperplaneGroup = new THREE.Group();
hyperplaneGroup.visible = false;
scene.add(hyperplaneGroup);

var hpGeom = new THREE.PlaneGeometry(10, 10);
var hpMat = new THREE.MeshBasicMaterial({{
    color: 0x00ffff, side: THREE.DoubleSide,
    transparent: true, opacity: 0.28, depthWrite: false,
}});
var hpPlane = new THREE.Mesh(hpGeom, hpMat);
hpPlane.position.z = DATA.hyperplane_z;
hyperplaneGroup.add(hpPlane);

var hpRingGeom = new THREE.TorusGeometry(4.2, 0.06, 12, 80);
var hpRingMat = new THREE.MeshBasicMaterial({{
    color: 0x00ffff, transparent: true, opacity: 0.5, depthTest: true,
}});
var hpRingMesh = new THREE.Mesh(hpRingGeom, hpRingMat);
hpRingMesh.position.z = DATA.hyperplane_z;
hyperplaneGroup.add(hpRingMesh);

// ============================================================
// 圖層 6 — 投影層
// ============================================================
var projectionGroup = new THREE.Group();
projectionGroup.visible = false;
scene.add(projectionGroup);

var projCircleGeom = new THREE.TorusGeometry(2.0, 0.05, 16, 80);
var projCircleMat = new THREE.MeshBasicMaterial({{
    color: 0x00ff88, transparent: true, opacity: 0.75, depthTest: true,
}});
var projCircle = new THREE.Mesh(projCircleGeom, projCircleMat);
projectionGroup.add(projCircle);

var projLinesGroup = new THREE.Group();
projectionGroup.add(projLinesGroup);

// ============================================================
// 座標軸 (淡色)
// ============================================================
var axisLen = 8;
var axisGroup = new THREE.Group();
function thinLine(s, e, c) {{ return lineMesh(s, e, c, 0.02, 0.25); }}
axisGroup.add(thinLine(new THREE.Vector3(-axisLen,0,0), new THREE.Vector3(axisLen,0,0), 0x444444));
axisGroup.add(thinLine(new THREE.Vector3(0,-axisLen,0), new THREE.Vector3(0,axisLen,0), 0x444444));
axisGroup.add(thinLine(new THREE.Vector3(0,0,-3), new THREE.Vector3(0,0,3), 0x444444));
scene.add(axisGroup);

// ============================================================
// 位置更新函式
// ============================================================

function setParticlePositions(morphT) {{
    var rArr = redGeom.attributes.position.array;
    var bArr = blueGeom.attributes.position.array;
    for (var pi = 0; pi < N; pi++) {{
        var r = redParticles[pi];
        var b = blueParticles[pi];
        var rx = lerp(r.linearPos.x, r.nonlinearPos.x, morphT);
        var ry = lerp(r.linearPos.y, r.nonlinearPos.y, morphT);
        var bx = lerp(b.linearPos.x, b.nonlinearPos.x, morphT);
        var by = lerp(b.linearPos.y, b.nonlinearPos.y, morphT);
        r.currentPos.set(rx, ry, r.currentPos.z);
        b.currentPos.set(bx, by, b.currentPos.z);
        var ri = pi * 3, bi = pi * 3;
        rArr[ri] = rx; rArr[ri+1] = ry; rArr[ri+2] = r.currentPos.z;
        bArr[bi] = bx; bArr[bi+1] = by; bArr[bi+2] = b.currentPos.z;
    }}
    redGeom.attributes.position.needsUpdate = true;
    blueGeom.attributes.position.needsUpdate = true;
}}

function applyKernelLift(kernelT) {{
    var rArr = redGeom.attributes.position.array;
    var bArr = blueGeom.attributes.position.array;
    for (var ki = 0; ki < N; ki++) {{
        var rz = lerp(0, redParticles[ki].kernelZNonlin, kernelT);
        var bz = lerp(0, blueParticles[ki].kernelZNonlin, kernelT);
        redParticles[ki].currentPos.z = rz;
        blueParticles[ki].currentPos.z = bz;
        rArr[ki*3+2] = rz;
        bArr[ki*3+2] = bz;
    }}
    redGeom.attributes.position.needsUpdate = true;
    blueGeom.attributes.position.needsUpdate = true;
}}

function updateGridLift(kernelT) {{
    for (var ui = 0; ui < gridNodes.length; ui++) {{
        var node = gridNodes[ui];
        var x = node.userData.baseX, y = node.userData.baseY;
        node.position.z = Math.exp(-(x*x + y*y)) * kernelT;
        node.material.opacity = 0.2 + kernelT * 0.35;
    }}
}}

function updateCamera3D(rotT) {{
    camera.position.set(rotT*5.5, rotT*5.5, 14 - rotT*5);
    camera.lookAt(0, 0, lerp(0, DATA.hyperplane_z, rotT));
}}

function updateProjectionLines(opacity) {{
    while (projLinesGroup.children.length > 0) projLinesGroup.remove(projLinesGroup[0]);
    if (opacity <= 0) return;
    var allParts = redParticles.concat(blueParticles);
    for (var li = 0; li < allParts.length; li += 4) {{
        var p = allParts[li];
        var top = p.currentPos.clone();
        var bot = p.currentPos.clone(); bot.z = 0;
        projLinesGroup.add(lineMesh(top, bot, 0x3355ff, 0.015, opacity*0.35));
    }}
}}

// ============================================================
// 可見性輔助
// ============================================================

function setDBOpacity(o) {{
    for (var di = 0; di < dbGroup.children.length; di++) {{
        var c = dbGroup.children[di];
        if (c.material) c.material.opacity = o * 0.9;
    }}
}}

function setLayerVisibility(group, visible) {{
    group.visible = visible;
}}

// ============================================================
// UI 更新 (全中文)
// ============================================================

var STATE_UI = {{
    0: {{
        name: '線性 SVM',
        color: '#0ff',
        formula: 'f(x) = w<sup>T</sup>x + b',
        html: '資料是 <span class="hl">完美線性可分</span> 的。<br>' +
              'SVM 找到 <span class="hl">最佳超平面</span>，<br>' +
              '最大化兩個類別之間的邊界距離。<br><br>' +
              '<span style="color:#0f0">▬</span> 決策邊界：w<sup>T</sup>x + b = 0<br>' +
              '<span class="wn">▬</span> 邊界線：w<sup>T</sup>x + b = ±1<br>' +
              '<span class="wn">◯</span> 支持向量（最近的點）',
    }},
    1: {{
        name: '非線性資料',
        color: '#f0f',
        formula: '在 ℝ² 空間中不存在線性分隔器',
        html: '資料 <span class="wn">無法</span> 用一條直線分開。<br>' +
              '紅色集中在中心，藍色在外圍環繞。<br>' +
              '沒有任何直線可以將它們分離。<br><br>' +
              '<span class="wn">？</span> SVM 該如何處理？<br>' +
              '<span style="color:#888;">→</span> <b>核方法</b> 將資料映射到<br>' +
              '更高維度的空間，使其<br>' +
              '<span class="hl">變得線性可分</span>！',
    }},
    2: {{
        name: '核方法 3D',
        color: '#ff0',
        formula: 'Φ(x₁,x₂) = (x₁, x₂, e<sup>−(x₁²+x₂²)</sup>)',
        html: '資料經由核函數 <span class="hl">提升到 3D</span>！<br>' +
              'K(x,y) = exp(−γ||x−y||²)<br><br>' +
              '<span style="color:#0ff">▭</span> 3D 空間中的分離超平面<br>' +
              '<span style="color:#0f0">○</span> 投影回 2D 的圓形邊界<br><br>' +
              '<span class="hl">在高維空間中，資料<br>變得線性可分！</span>',
    }},
}};

function updateUIOverlay() {{
    var ui = STATE_UI[currentState];
    document.getElementById('state-name').textContent = ui.name;
    document.getElementById('state-name').style.color = ui.color;
    document.getElementById('formula').innerHTML = ui.formula;
    document.getElementById('explanation').innerHTML = ui.html;
    document.getElementById('note-banner').style.display = currentState === STATE.KERNEL_3D ? 'block' : 'none';

    ['btn-linear','btn-nonlinear','btn-kernel'].forEach(function(id) {{
        document.getElementById(id).classList.remove('active');
    }});
    var activeMap = ['btn-linear','btn-nonlinear','btn-kernel'];
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
// 轉換引擎
// ============================================================

function requestTransition(toState) {{
    if (targetState === toState && transitionProgress < 1 && transitionProgress > 0) return;
    if (currentState === toState) return;
    targetState = toState;
    transitionProgress = 0;
    transitionStartMs = performance.now();
}}

function tickTransition(now) {{
    if (targetState === currentState) return;
    var elapsed = now - transitionStartMs;
    transitionProgress = Math.min(elapsed / T_DURATION, 1.0);
    var t = easeInOutCubic(transitionProgress);
    var from = currentState;

    // LINEAR → NONLINEAR
    if (from === STATE.LINEAR && targetState === STATE.NONLINEAR) {{
        setParticlePositions(t);
        setDBOpacity(1 - t);
        refreshSVIndicators(0.85 * (1 - t));
    }}

    // NONLINEAR → KERNEL_3D
    if (from === STATE.NONLINEAR && targetState === STATE.KERNEL_3D) {{
        var liftT = smoothstep(0.0, 0.35, t);
        applyKernelLift(liftT);
        var showT = smoothstep(0.2, 1.0, t);
        kernelGridGroup.visible = showT > 0.05;
        updateGridLift(Math.min(showT * 1.4, 1.0));
        hyperplaneGroup.visible = showT > 0.25;
        hpMat.opacity = 0.28 * smoothstep(0.25, 0.5, t);
        hpRingMat.opacity = 0.5 * smoothstep(0.25, 0.5, t);
        projectionGroup.visible = showT > 0.35;
        projCircleMat.opacity = 0.75 * smoothstep(0.35, 0.6, t);
        updateProjectionLines(smoothstep(0.35, 0.7, t));
        updateCamera3D(showT);
        controls.enabled = showT > 0.5;
        document.getElementById('note-banner').style.display = showT > 0.4 ? 'block' : 'none';
    }}

    // LINEAR → KERNEL_3D (串聯轉換)
    if (from === STATE.LINEAR && targetState === STATE.KERNEL_3D) {{
        var phase1End = 0.35, phase2Start = 0.25;
        if (t <= phase1End) {{
            var pt = Math.min(t/phase1End, 1.0);
            setParticlePositions(pt);
            setDBOpacity(1 - pt);
            refreshSVIndicators(0.85 * (1 - pt));
        }} else {{
            setParticlePositions(1);
            setDBOpacity(0);
            refreshSVIndicators(0);
        }}
        var liftT2 = smoothstep(phase2Start, 0.65, t);
        applyKernelLift(liftT2);
        kernelGridGroup.visible = liftT2 > 0.05;
        updateGridLift(Math.min(liftT2 * 1.4, 1.0));
        var showT2 = smoothstep(phase2Start + 0.05, 1.0, t);
        hyperplaneGroup.visible = showT2 > 0.2;
        hpMat.opacity = 0.28 * smoothstep(0.2, 0.45, showT2);
        hpRingMat.opacity = 0.5 * smoothstep(0.2, 0.45, showT2);
        projectionGroup.visible = showT2 > 0.3;
        projCircleMat.opacity = 0.75 * smoothstep(0.3, 0.55, showT2);
        updateProjectionLines(smoothstep(0.3, 0.65, showT2));
        updateCamera3D(showT2);
        controls.enabled = showT2 > 0.45;
        document.getElementById('note-banner').style.display = showT2 > 0.35 ? 'block' : 'none';
    }}

    // 轉換完成
    if (transitionProgress >= 1.0) {{
        currentState = targetState;
        transitionProgress = 0;
        updateCameraForState();
        updateUIOverlay();
    }}
}}

// ============================================================
// 重置到線性狀態
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
// 按鈕事件
// ============================================================
document.getElementById('btn-linear').addEventListener('click', function() {{
    resetToLinear();
}});

document.getElementById('btn-nonlinear').addEventListener('click', function() {{
    if (currentState === STATE.NONLINEAR) return;
    if (currentState === STATE.KERNEL_3D) {{
        resetToLinear();
        setTimeout(function() {{ requestTransition(STATE.NONLINEAR); }}, 80);
    }} else {{
        requestTransition(STATE.NONLINEAR);
    }}
}});

document.getElementById('btn-kernel').addEventListener('click', function() {{
    if (currentState === STATE.KERNEL_3D) return;
    requestTransition(STATE.KERNEL_3D);
}});

// ============================================================
// FPS 計數器
// ============================================================
var frameCount = 0;
var lastFpsTime = performance.now();
var fpsEl = document.getElementById('fps-counter');

function updateFPS(now) {{
    frameCount++;
    if (now - lastFpsTime >= 1000) {{
        var fps = Math.round(frameCount / ((now - lastFpsTime) / 1000));
        fpsEl.textContent = 'FPS: ' + fps;
        frameCount = 0;
        lastFpsTime = now;
    }}
}}

// ============================================================
// 主動畫迴圈
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
    refreshSVIndicators(currentState === STATE.LINEAR ? 0.85 : 0);
    controls.update();
    renderer.render(scene, camera);
    updateFPS(timestamp);
}}

// ============================================================
// 視窗大小調整
// ============================================================
window.addEventListener('resize', function() {{
    var w2 = window.innerWidth, h2 = window.innerHeight;
    camera.aspect = w2 / h2;
    camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
}});

// ============================================================
// 初始化 — 顯示 UI 並開始動畫
// ============================================================
try {{
    setParticlePositions(0);
    updateUIOverlay();
    updateCameraForState();
    hideLoading();
    showUI();
    requestAnimationFrame(animate);
}} catch(e) {{
    showError('場景初始化失敗：' + e.message);
    throw e;
}}

}})(); // IIFE 結束
</script>
</body>
</html>"""


# ============================================================
# STREAMLIT 進入點 (中文)
# ============================================================

def main():
    st.set_page_config(
        page_title="SVM 核方法 — 3D 互動視覺化",
        page_icon="🔮",
        layout="wide",
    )
    st.title("🔮 SVM 核方法：3D 互動視覺化")
    st.markdown(
        """
        **教學視覺化**：展示支持向量機 (SVM) 如何透過
        **核方法 (Kernel Trick)** 將非線性可分的資料映射到
        高維空間中，使其變得線性可分。

        使用下方 3D 畫面中的按鈕逐步切換狀態：
        **線性 SVM** → **非線性資料** → **核方法 3D**。
        在 3D 狀態下，可拖曳旋轉、滾輪縮放、右鍵平移。
        """
    )

    data = generate_datasets()
    data_json = to_json_compact(data)
    html = build_html(data_json)
    components.html(html, height=750, scrolling=True)


if __name__ == "__main__":
    main()
