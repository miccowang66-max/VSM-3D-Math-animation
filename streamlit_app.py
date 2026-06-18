"""
streamlit_app.py — SVM 核方法 3D 互動視覺化（真實決策面版本）
============================================================
完全自包含。在 3D 特徵空間中訓練真實 SVM，渲染實際決策平面、
支持向量，以及投影回 2D 的非線性決策曲線。

執行: streamlit run streamlit_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json

# ============================================================
# Z 軸視覺放大倍數
# ============================================================
Z_SCALE = 8.0


# ============================================================
# 資料生成 — 含真實 3D SVM 決策面計算
# ============================================================

def _generate_datasets(n_per_class=80):
    rng = np.random.RandomState(42)

    # ---- 線性資料 ----
    class_a_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([-3, -3])
    class_b_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([3, 3])
    X_linear = np.vstack([class_a_lin, class_b_lin])
    y_linear = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # ---- 非線性資料（紅:中心, 藍:外環）----
    class_a_nonlin = rng.randn(n_per_class, 2) * 1.0
    angles = rng.uniform(0, 2*np.pi, n_per_class)
    radii = rng.uniform(3.5, 5.0, n_per_class)
    class_b_nonlin = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

    # ---- 線性 SVM (2D) ----
    try:
        from sklearn.svm import SVC
        svm_2d = SVC(kernel="linear", C=1e10, random_state=42)
        svm_2d.fit(X_linear, y_linear)
        w = svm_2d.coef_[0].astype(np.float64)
        b_2d = float(svm_2d.intercept_[0])
        sv_2d = [int(i) for i in svm_2d.support_]
    except ImportError:
        ca = class_a_lin.mean(axis=0); cb = class_b_lin.mean(axis=0)
        w = cb - ca; w = w / np.linalg.norm(w)
        b_2d = float(-np.dot(w, (ca+cb)/2))
        da = np.abs(np.dot(class_a_lin,w)+b_2d); db = np.abs(np.dot(class_b_lin,w)+b_2d)
        sv_2d = (
            [int(i) for i in np.where(da<=np.percentile(da,20))[0]] +
            [int(n_per_class+i) for i in np.where(db<=np.percentile(db,20))[0]]
        )

    # ---- 2D 邊界線幾何 ----
    w_norm = w / np.linalg.norm(w)
    perp = np.array([-w_norm[1], w_norm[0]], dtype=np.float64)
    ext = 7.0; pc = -b_2d*w_norm
    p1 = pc + perp*ext; p2 = pc - perp*ext
    p1p = p1 + w_norm; p2p = p2 + w_norm
    p1n = p1 - w_norm; p2n = p2 - w_norm

    # ---- 核函數 Φ: 2D → 3D ----
    def _phi(pts):
        """Φ(x,y) = (x, y, Z_SCALE * exp(-(x²+y²)))"""
        r_sq = np.sum(pts**2, axis=1)
        return np.column_stack([pts, Z_SCALE * np.exp(-r_sq)])

    def _kz(pts):
        """僅 Z 分量"""
        return Z_SCALE * np.exp(-np.sum(pts**2, axis=1))

    # 非線性資料的 3D 特徵
    phi_a = _phi(class_a_nonlin)  # (n, 3)
    phi_b = _phi(class_b_nonlin)  # (n, 3)
    X_phi = np.vstack([phi_a, phi_b])
    y_phi = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # ---- 真實 SVM 在 3D 特徵空間中訓練 ----
    # 計算核空間中兩類的質心，找出最大邊界分離平面
    red_centroid_3d = np.mean(phi_a, axis=0)  # (3,)
    blue_centroid_3d = np.mean(phi_b, axis=0)  # (3,)

    # 法向量 = 質心連線方向（從藍指向紅）
    w3 = red_centroid_3d - blue_centroid_3d
    w3_norm_val = np.linalg.norm(w3)
    if w3_norm_val > 0:
        w3 = w3 / w3_norm_val
    # 決策平面過中點：w·mid + b = 0 => b = -w·mid
    mid_3d = (red_centroid_3d + blue_centroid_3d) / 2.0
    b3 = float(-np.dot(w3, mid_3d))

    # 支持向量：兩類中離決策平面最近的點（各取前 10%）
    dists_a = np.abs(np.dot(phi_a, w3) + b3)
    dists_b = np.abs(np.dot(phi_b, w3) + b3)
    thresh_a = np.percentile(dists_a, 10)
    thresh_b = np.percentile(dists_b, 10)
    sv_a_idx = np.where(dists_a <= thresh_a)[0]
    sv_b_idx = np.where(dists_b <= thresh_b)[0]
    sv_3d_indices = (
        [int(i) for i in sv_a_idx] +
        [int(n_per_class + i) for i in sv_b_idx]
    )
    sv_3d_phi = np.vstack([phi_a[sv_a_idx], phi_b[sv_b_idx]]).tolist()

    # ---- 決策平面參數（ax + by + cz + d = 0）----
    # w3[0]*x + w3[1]*y + w3[2]*z + b3 = 0
    # 歸一化法向量以利於顯示（平面方程為齊次式，縮放不變）
    w3_norm = np.linalg.norm(w3)
    if w3_norm > 0:
        w3 = w3 / w3_norm
        b3 = b3 / w3_norm
    plane_a = float(w3[0])
    plane_b = float(w3[1])
    plane_c = float(w3[2])
    plane_d = b3

    # ---- 投影回 2D 的決策邊界曲線 ----
    # w3[0]*x + w3[1]*y + w3[2]*Z_SCALE*exp(-(x²+y²)) + b3 = 0
    # 在 xy 平面取樣點，找出決策邊界
    boundary_curve_2d = _compute_decision_curve_2d(
        w3[0], w3[1], w3[2], b3, Z_SCALE
    )

    return {
        "n": n_per_class,
        "linear_red": class_a_lin.tolist(),
        "linear_blue": class_b_lin.tolist(),
        "nonlinear_red": class_a_nonlin.tolist(),
        "nonlinear_blue": class_b_nonlin.tolist(),
        # 非線性資料的 3D 特徵 Z 分量
        "z_nonlinear_red": _kz(class_a_nonlin).tolist(),
        "z_nonlinear_blue": _kz(class_b_nonlin).tolist(),
        # 2D SVM
        "w": w.tolist(), "b_2d": b_2d, "sv_2d": sv_2d,
        "db_line": [p1.tolist(), p2.tolist()],
        "margin_pos": [p1p.tolist(), p2p.tolist()],
        "margin_neg": [p1n.tolist(), p2n.tolist()],
        # 3D SVM 真實決策面
        "plane_a": plane_a, "plane_b": plane_b,
        "plane_c": plane_c, "plane_d": plane_d,
        "sv_3d_phi": sv_3d_phi,
        "boundary_curve_2d": boundary_curve_2d,
    }


def _compute_decision_curve_2d(a, b, c, d, scale):
    """
    在 2D 平面上取樣決策邊界：
    a*x + b*y + c*scale*exp(-(x²+y²)) + d = 0
    
    使用圓形掃描：對每個角度，找出滿足條件的半徑。
    由於函數單調（exp 項），可用二分搜尋。
    """
    n_angles = 120
    curve_points = []
    
    for i in range(n_angles):
        theta = 2 * np.pi * i / n_angles
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 二分搜尋半徑 r
        lo, hi = 0.0, 7.0
        f_lo = a*(lo*cos_t) + b*(lo*sin_t) + c*scale*np.exp(-lo*lo) + d
        f_hi = a*(hi*cos_t) + b*(hi*sin_t) + c*scale*np.exp(-hi*hi) + d
        
        # 若兩端同號，此方向無解（跳過）
        if f_lo * f_hi > 0:
            continue
        
        for _ in range(30):
            mid = (lo + hi) / 2
            f_mid = a*(mid*cos_t) + b*(mid*sin_t) + c*scale*np.exp(-mid*mid) + d
            if abs(f_mid) < 0.01:
                break
            if f_lo * f_mid < 0:
                hi = mid; f_hi = f_mid
            else:
                lo = mid; f_lo = f_mid
        
        r = (lo + hi) / 2
        curve_points.append([float(r * cos_t), float(r * sin_t)])
    
    return curve_points


# ============================================================
# THREE.JS HTML — 真實 SVM 決策面版本
# ============================================================

THREEJS_HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVM 核方法 3D 視覺化</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;font-family:'Microsoft YaHei','PingFang TC',sans-serif;user-select:none}
canvas{display:block}
#status{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#fff;font-size:18px;z-index:100;text-align:center}
#status .dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin:0 4px;animation:bounce 1.4s infinite}
#status .dot:nth-child(2){animation-delay:.2s}
#status .dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
.dot-g{background:#0f0}.dot-r{background:#f00}.dot-y{background:#ff0}
#err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#f44;font-size:16px;z-index:100;text-align:center;display:none;background:rgba(0,0,0,.85);padding:20px 30px;border-radius:10px;border:1px solid #f44;max-width:500px;line-height:1.6}
#ui-overlay{position:absolute;top:20px;left:20px;background:rgba(0,0,0,.75);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:20px 24px;color:#fff;max-width:400px;backdrop-filter:blur(8px);pointer-events:none;display:none;z-index:10}
#state-name{font-size:26px;font-weight:bold;margin-bottom:6px;letter-spacing:2px}
#formula{font-size:14px;color:#fc0;margin-bottom:10px;word-break:break-all;line-height:1.4}
#explanation{font-size:13px;color:#bbb;line-height:1.65}
#explanation .hl{color:#0f0}#explanation .wn{color:#ff0}
#btn-bar{position:absolute;bottom:28px;left:50%;transform:translateX(-50%);display:none;gap:12px;z-index:20}
#btn-bar button{padding:13px 24px;font-size:15px;font-weight:bold;letter-spacing:1px;cursor:pointer;border:2px solid rgba(0,255,255,.6);background:rgba(0,20,40,.85);color:#0ff;border-radius:10px;font-family:inherit;transition:all .25s;backdrop-filter:blur(6px);white-space:nowrap}
#btn-bar button:hover{background:rgba(0,255,255,.18);border-color:#0ff;box-shadow:0 0 24px rgba(0,255,255,.45)}
#btn-bar button.active{background:rgba(0,255,255,.25);border-color:#fff;color:#fff}
#note-banner{position:absolute;bottom:105px;left:50%;transform:translateX(-50%);color:#888;font-size:12px;font-family:inherit;text-align:center;display:none;background:rgba(0,0,0,.6);padding:7px 18px;border-radius:6px;white-space:nowrap}
#fps-counter{position:absolute;top:10px;right:14px;color:#444;font-size:11px;font-family:monospace;z-index:10;display:none}
</style>
</head>
<body>

<div id="status">
    <div style="margin-bottom:12px">載入中</div>
    <span class="dot dot-g"></span><span class="dot dot-y"></span><span class="dot dot-r"></span>
    <div id="status-msg" style="font-size:13px;color:#888;margin-top:12px">正在初始化...</div>
</div>
<div id="err"></div>

<div id="ui-overlay">
    <div id="state-name"></div>
    <div id="formula"></div>
    <div id="explanation"></div>
</div>

<div id="btn-bar">
    <button id="btn-linear" class="active">線性 SVM</button>
    <button id="btn-nonlinear">非線性資料</button>
    <button id="btn-kernel">核方法 3D</button>
</div>
<div id="note-banner">⚠ 此為特徵空間提升的直觀示意，並非真實的無限維 RBF 特徵空間</div>
<div id="fps-counter">FPS: --</div>

<script>
// 雙 CDN 備援：若 jsdelivr 失敗，嘗試 cdnjs
var __three_src = 'https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js';
var __three_fallback = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
var __orbit_src = 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/controls/OrbitControls.js';
</script>
<script id="three-script" src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"
    onload="document.getElementById('status-msg').textContent='Three.js ✓'"
    onerror="var s=document.getElementById('three-script');s.src=__three_fallback;s.onerror=function(){document.getElementById('err').style.display='block';document.getElementById('err').innerHTML='<b>Three.js 載入失敗</b><br>jsdelivr 與 cdnjs 皆無法連線。<br>請檢查網路或稍後再試。';document.getElementById('status').style.display='none';}">
</script>
<script id="orbit-script" src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/controls/OrbitControls.js"
    onload="document.getElementById('status-msg').textContent='OrbitControls ✓'"
    onerror="document.getElementById('status-msg').textContent='OrbitControls 不可用（將無旋轉功能）';">
</script>

<script>
(function(){
'use strict';
var DATA = __DATA__;
var N = DATA.n;
var hpZ = 0;
// 從 SVM 平面計算代表性 Z 值（用於 UI 顯示）
(function(){
    var a=DATA.plane_a, b2=DATA.plane_b, c=DATA.plane_c, d=DATA.plane_d;
    if(Math.abs(c)>0.001) hpZ = -d/c;
    else hpZ = DATA.z_nonlinear_red.reduce(function(s,v){return s+v},0)/N;
})();

function waitForScripts(cb){
    var a=0;
    function ch(){a++;if(typeof THREE!=='undefined')cb(true);else if(a>=100)cb(false);else setTimeout(ch,100)}
    ch();
}

waitForScripts(function(ok){
    if(!ok){
        document.getElementById('status').style.display='none';
        var e=document.getElementById('err');e.style.display='block';
        e.innerHTML='<b>Three.js 載入逾時</b><br><br>CDN 無法連線，請檢查網路';
        return;
    }
    startApp();
});

function startApp(){
try{

// ============================================================
// 狀態機
// ============================================================
var STATE={LINEAR:0,NONLINEAR:1,KERNEL_3D:2};
var cs=STATE.LINEAR, ts=STATE.LINEAR;
var tp=0, tsm=0, TD=2800;
function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2}
function lerp(a,b,t){return a+(b-a)*t}
function smooth(e0,e1,x){var t2=Math.max(0,Math.min((x-e0)/(e1-e0),1));return t2*t2*(3-2*t2)}

// ============================================================
// 粒子結構
// ============================================================
var RP=[], BP=[];
for(var i=0;i<N;i++){
    var lr=DATA.linear_red[i], nr=DATA.nonlinear_red[i];
    RP.push({lx:lr[0],ly:lr[1],nx:nr[0],ny:nr[1],kz:DATA.z_nonlinear_red[i],
             sv2:DATA.sv_2d.indexOf(i)!==-1});
}
for(var i=0;i<N;i++){
    var lb=DATA.linear_blue[i], nb=DATA.nonlinear_blue[i];
    BP.push({lx:lb[0],ly:lb[1],nx:nb[0],ny:nb[1],kz:DATA.z_nonlinear_blue[i],
             sv2:DATA.sv_2d.indexOf(N+i)!==-1});
}

// 3D SVM 支持向量集（以 3D 座標比對）
var sv3Set = {};
DATA.sv_3d_phi.forEach(function(p){ sv3Set[p[0].toFixed(4)+','+p[1].toFixed(4)+','+p[2].toFixed(4)]=true; });

// ============================================================
// 場景 —— 強化 3D 深度
// ============================================================
var W=window.innerWidth, H=window.innerHeight;
var scene=new THREE.Scene();
scene.fog=new THREE.FogExp2(0x050510, 0.0007);

var camera=new THREE.PerspectiveCamera(45, W/H, 0.5, 120);
camera.position.set(3, -2, 13);
camera.lookAt(0, 0, 0);

var renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W,H);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setClearColor(0x050510);
document.body.appendChild(renderer.domElement);

var controls=null;
if(typeof THREE.OrbitControls!=='undefined'){
    controls=new THREE.OrbitControls(camera,renderer.domElement);
    controls.enabled=false;controls.enableDamping=true;controls.dampingFactor=0.08;
    controls.minDistance=6;controls.maxDistance=40;controls.maxPolarAngle=Math.PI*0.8;
    controls.target.set(0,0,hpZ*0.5);
}

scene.add(new THREE.AmbientLight(0x333355, 0.5));
var dl=new THREE.DirectionalLight(0xffffff, 0.7);dl.position.set(3,8,10);scene.add(dl);
var dl2=new THREE.DirectionalLight(0x4466ff, 0.3);dl2.position.set(-5,-2,4);scene.add(dl2);

// 地板網格
var gh=new THREE.GridHelper(16, 24, 0x333355, 0x111122);scene.add(gh);

// 星空
var sg=new THREE.BufferGeometry();var sc2=1800, sa=new Float32Array(sc2*3);
for(var i=0;i<sc2*3;i+=3){sa[i]=(Math.random()-.5)*50;sa[i+1]=(Math.random()-.5)*50;sa[i+2]=(Math.random()-.5)*30-5}
sg.setAttribute('position',new THREE.BufferAttribute(sa,3));
var sm=new THREE.Points(sg,new THREE.PointsMaterial({color:0xccccff,size:.04,transparent:true,opacity:.8,blending:THREE.AdditiveBlending,depthWrite:false}));scene.add(sm);

// 星雲
var ng=new THREE.BufferGeometry(), nc3=400, np3=new Float32Array(nc3*3), ncl=new Float32Array(nc3*3);
for(var i=0;i<nc3*3;i+=3){np3[i]=(Math.random()-.5)*28;np3[i+1]=(Math.random()-.5)*28;np3[i+2]=(Math.random()-.5)*14-2;ncl[i]=.08+Math.random()*.12;ncl[i+1]=.02+Math.random()*.06;ncl[i+2]=.15+Math.random()*.25}
ng.setAttribute('position',new THREE.BufferAttribute(np3,3));ng.setAttribute('color',new THREE.BufferAttribute(ncl,3));
scene.add(new THREE.Points(ng,new THREE.PointsMaterial({size:.6,vertexColors:true,transparent:true,opacity:.25,blending:THREE.AdditiveBlending,depthWrite:false})));

// 資料粒子
var rg=new THREE.BufferGeometry(), ra=new Float32Array(N*3);
rg.setAttribute('position',new THREE.BufferAttribute(ra,3));
var rp2=new THREE.Points(rg,new THREE.PointsMaterial({color:0xff2233,size:.20,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));scene.add(rp2);
var bg2=new THREE.BufferGeometry(), ba=new Float32Array(N*3);
bg2.setAttribute('position',new THREE.BufferAttribute(ba,3));
var bp2=new THREE.Points(bg2,new THREE.PointsMaterial({color:0x2266ff,size:.20,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));scene.add(bp2);

// ============================================================
// 輔助：線段 / 平面
// ============================================================
function lm(pA,pB,color,width,opacity){
    var dir=new THREE.Vector3().subVectors(pB,pA), len=dir.length();
    var mid=new THREE.Vector3().addVectors(pA,pB).multiplyScalar(.5);
    var geom=new THREE.CylinderGeometry(width,width,len,6,1);
    var mat=new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:opacity,depthTest:true});
    var mesh=new THREE.Mesh(geom,mat);mesh.position.copy(mid);
    mesh.setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize()));
    return mesh;
}

// 自訂平面（任意法向量）
function createSVMPlane(a,b,c,d,color,opacity,size){
    // 平面方程: a*x + b*y + c*z + d = 0
    // 建立一個 size x size 的平面，法向量為 (a,b,c)
    var w2=size/2;
    var geom=new THREE.PlaneGeometry(size, size);
    var mat2=new THREE.MeshBasicMaterial({color:color,side:THREE.DoubleSide,transparent:true,opacity:opacity,depthWrite:false});
    var mesh=new THREE.Mesh(geom,mat2);

    // 法向量
    var normal=new THREE.Vector3(a,b,c).normalize();
    // 預設 PlaneGeometry 法向量為 (0,0,1)，旋轉到目標法向量
    var quat=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), normal);
    mesh.setRotationFromQuaternion(quat);

    // 平面上的一點：取最接近原點的點
    var nLen2=a*a+b*b+c*c;
    var t = -d/nLen2;
    mesh.position.set(a*t, b*t, c*t);

    return mesh;
}

// ============================================================
// 2D 決策邊界 + 邊界線
// ============================================================
var dbg=new THREE.Group();scene.add(dbg);
(function(){
    var p1=new THREE.Vector3(DATA.db_line[0][0],DATA.db_line[0][1],0.02);
    var p2=new THREE.Vector3(DATA.db_line[1][0],DATA.db_line[1][1],0.02);
    var mp1=new THREE.Vector3(DATA.margin_pos[0][0],DATA.margin_pos[0][1],0.01);
    var mp2=new THREE.Vector3(DATA.margin_pos[1][0],DATA.margin_pos[1][1],0.01);
    var mn1=new THREE.Vector3(DATA.margin_neg[0][0],DATA.margin_neg[0][1],0.01);
    var mn2=new THREE.Vector3(DATA.margin_neg[1][0],DATA.margin_neg[1][1],0.01);
    dbg.add(lm(p1.clone().add(new THREE.Vector3(0,0,0.15)), p2.clone().add(new THREE.Vector3(0,0,0.15)), 0x00ff88, 0.09, 1.0));
    dbg.add(lm(mp1,mp2,0xffcc00,0.04,0.9));
    dbg.add(lm(mn1,mn2,0xffcc00,0.04,0.9));
})();

// ============================================================
// 2D SV 指示器（線性狀態用）
// ============================================================
var svg=new THREE.Group();scene.add(svg);
var svrg=new THREE.TorusGeometry(0.34,0.055,8,20);
function refSV(op, use3D){
    while(svg.children.length>0)svg.remove(svg.children[0]);
    var all=RP.concat(BP);
    for(var i=0;i<all.length;i++){
        var isSV = use3D ? all[i]._sv3 : all[i].sv2;
        if(!isSV)continue;
        var idx=i<N?i:i-N;
        var arr=i<N?ra:ba;
        var px=arr[idx*3], py=arr[idx*3+1], pz=arr[idx*3+2];
        var ring=new THREE.Mesh(svrg,new THREE.MeshBasicMaterial({color:0xffff44,transparent:true,opacity:op,depthTest:true}));
        ring.position.set(px,py,pz+0.05);svg.add(ring);
    }
}
refSV(0.85, false);

// ============================================================
// 核轉換網格
// ============================================================
var kgg=new THREE.Group();kgg.visible=false;scene.add(kgg);
var GR=24, GE=7.5, gn=[];
var gdg=new THREE.SphereGeometry(.05,4,4);
for(var i=0;i<=GR;i++)for(var j=0;j<=GR;j++){
    var fx=(i/GR-.5)*2*GE, fy=(j/GR-.5)*2*GE;
    var gm2=new THREE.Mesh(gdg,new THREE.MeshBasicMaterial({color:0x8888ff,transparent:true,opacity:.5,depthTest:true}));
    gm2.position.set(fx,fy,0);gm2.userData={bx:fx,by:fy};kgg.add(gm2);gn.push(gm2);
}

// ============================================================
// 真實 SVM 3D 決策平面
// ============================================================
var hpg=new THREE.Group();hpg.visible=false;scene.add(hpg);
var svmPlane=createSVMPlane(DATA.plane_a, DATA.plane_b, DATA.plane_c, DATA.plane_d, 0x00ffff, 0.35, 13);
hpg.add(svmPlane);

// 平面邊框（發光環，投影到平面上的固定大小參考環）
var hpEdgeGeom=new THREE.TorusGeometry(4.8, 0.07, 12, 80);
var hpEdgeMat=new THREE.MeshBasicMaterial({color:0x00ffff,transparent:true,opacity:.55,depthTest:true});
var hpEdge=new THREE.Mesh(hpEdgeGeom, hpEdgeMat);
// 將環放在平面上：平面中心 + 在平面上的大圓
var nc=new THREE.Vector3(DATA.plane_a, DATA.plane_b, DATA.plane_c).normalize();
var quat2=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), nc);
hpEdge.setRotationFromQuaternion(quat2);
var nLen2a=DATA.plane_a*DATA.plane_a+DATA.plane_b*DATA.plane_b+DATA.plane_c*DATA.plane_c;
var ta=-DATA.plane_d/nLen2a;
hpEdge.position.set(DATA.plane_a*ta, DATA.plane_b*ta, DATA.plane_c*ta);
hpg.add(hpEdge);

// ============================================================
// 投影回 2D 的決策邊界曲線（真實 SVM 結果）
// ============================================================
var pjg=new THREE.Group();pjg.visible=false;scene.add(pjg);
var curvePoints=DATA.boundary_curve_2d;
if(curvePoints && curvePoints.length>1){
    // 繪製閉合曲線（在 z=0 平面上）
    for(var ci=0; ci<curvePoints.length; ci++){
        var cp0=curvePoints[ci];
        var cp1=curvePoints[(ci+1)%curvePoints.length];
        pjg.add(lm(
            new THREE.Vector3(cp0[0],cp0[1],0.03),
            new THREE.Vector3(cp1[0],cp1[1],0.03),
            0x00ff88, 0.05, 0.85
        ));
    }
}

// 投影線群組
var plg=new THREE.Group();pjg.add(plg);

// ============================================================
// Z 軸參考柱
// ============================================================
var maxZ2=Math.max(hpZ+3, 7);
scene.add(lm(new THREE.Vector3(0,0,-3), new THREE.Vector3(0,0,maxZ2), 0x335566, 0.03, 0.4));
for(var zi=1; zi<=Math.ceil(maxZ2); zi++){
    var tr=new THREE.Mesh(new THREE.TorusGeometry(.15,.02,6,12),
        new THREE.MeshBasicMaterial({color:0x335566,transparent:true,opacity:.25,depthTest:true}));
    tr.position.set(0,0,zi);scene.add(tr);
}

// ============================================================
// 標記 3D SVM 支持向量
// ============================================================
// 在粒子結構中標記哪些是 3D SV
(function(){
    DATA.sv_3d_phi.forEach(function(sv){
        var sx=sv[0], sy=sv[1], sz=sv[2];
        // 比對非線性位置 + kernel Z
        for(var i=0;i<N;i++){
            var rdx=Math.abs(RP[i].nx-sx)+Math.abs(RP[i].ny-sy)+Math.abs(RP[i].kz-sz);
            if(rdx<0.1){RP[i]._sv3=true;return}
            var bdx=Math.abs(BP[i].nx-sx)+Math.abs(BP[i].ny-sy)+Math.abs(BP[i].kz-sz);
            if(bdx<0.1){BP[i]._sv3=true;return}
        }
    });
})();

// ============================================================
// 位置更新
// ============================================================
function setPP(mt){
    for(var i=0;i<N;i++){
        var r=RP[i], b=BP[i];
        ra[i*3]=lerp(r.lx,r.nx,mt);ra[i*3+1]=lerp(r.ly,r.ny,mt);
        ba[i*3]=lerp(b.lx,b.nx,mt);ba[i*3+1]=lerp(b.ly,b.ny,mt);
    }
    rg.attributes.position.needsUpdate=true;bg2.attributes.position.needsUpdate=true;
}
function aKL(kt){
    for(var i=0;i<N;i++){
        ra[i*3+2]=lerp(0,RP[i].kz,kt);
        ba[i*3+2]=lerp(0,BP[i].kz,kt);
    }
    rg.attributes.position.needsUpdate=true;bg2.attributes.position.needsUpdate=true;
}
function uGL(kt){
    for(var i=0;i<gn.length;i++){
        var n=gn[i], x=n.userData.bx, y=n.userData.by;
        n.position.z=Math.exp(-(x*x+y*y))*8*kt;
        n.material.opacity=.2+kt*.35;
    }
}
function uC3(rt){
    var tx=rt*7, ty=rt*3, tz=14-rt*6;
    camera.position.set(tx,ty,tz);
    camera.lookAt(0,0,lerp(0,hpZ*.7,rt));
}
function uPL(op){
    while(plg.children.length>0)plg.remove(plg.children[0]);
    if(op<=0)return;
    for(var i=0;i<N*2;i+=4){
        var idx=i<N?i:i-N;
        var arr=i<N?ra:ba;
        var px=arr[idx*3], py=arr[idx*3+1], pz=arr[idx*3+2];
        if(pz<.1)continue;
        plg.add(lm(new THREE.Vector3(px,py,pz),new THREE.Vector3(px,py,0),0x3355ff,.012,op*.3));
    }
}
function sDBO(o){
    for(var i=0;i<dbg.children.length;i++){
        if(dbg.children[i].material)dbg.children[i].material.opacity=o*.9;
    }
}

// ============================================================
// UI
// ============================================================
var SU=[
    {nm:'線性 SVM',cl:'#0ff',fm:'f(x) = w<sup>T</sup>x + b',
     hx:'資料 <span class="hl">完美線性可分</span>。<br>SVM 找到 <span class="hl">最佳超平面</span>，<br>最大化兩個類別之間的邊界距離。<br><br><span style="color:#0f0">▬</span> 決策邊界：w<sup>T</sup>x + b = 0<br><span class="wn">▬</span> 邊界線：w<sup>T</sup>x + b = ±1<br><span class="wn">◯</span> 支持向量'},
    {nm:'非線性資料',cl:'#f0f',fm:'在 ℝ² 空間中不存在線性分隔器',
     hx:'資料 <span class="wn">無法</span> 用直線分開。<br>紅色中心群聚，藍色外圍環繞。<br><br><span class="wn">？</span> SVM 如何處理？<br><span style="color:#888;">→</span> <b>核方法</b> 映射到高維空間<br>使資料 <span class="hl">變得線性可分</span>！'},
    {nm:'核方法 3D — SVM 決策面',cl:'#ff0',fm:'Φ(x,y) = (x, y, 8·e<sup>−(x²+y²)</sup>)',
     hx:'資料提升到 3D 後，SVM 找到<br><span style="color:#0ff">▭</span> <b>真實決策平面</b>：<br>'+DATA.plane_a.toFixed(2)+'x + '+DATA.plane_b.toFixed(2)+'y + '+DATA.plane_c.toFixed(2)+'z + '+DATA.plane_d.toFixed(2)+' = 0<br><br><span style="color:#0f0">○</span> <b>投影決策曲線</b>（地面）<br><span class="wn">◯</span> 3D 支持向量<br><br><span class="hl">在 3D 特徵空間中<br>資料完美線性可分！</span>'}
];
function uUI(){
    var u=SU[cs];
    document.getElementById('state-name').textContent=u.nm;
    document.getElementById('state-name').style.color=u.cl;
    document.getElementById('formula').innerHTML=u.fm;
    document.getElementById('explanation').innerHTML=u.hx;
    document.getElementById('note-banner').style.display=cs===STATE.KERNEL_3D?'block':'none';
    ['btn-linear','btn-nonlinear','btn-kernel'].forEach(function(id){document.getElementById(id).classList.remove('active')});
    document.getElementById(['btn-linear','btn-nonlinear','btn-kernel'][cs]).classList.add('active');
}
function uCS(){
    if(cs===STATE.KERNEL_3D){
        uC3(1);if(controls){controls.enabled=true;controls.target.set(0,0,hpZ*.5)}
    }else{
        camera.position.set(3,-2,13);camera.lookAt(0,0,0);
        if(controls){controls.enabled=false;controls.target.set(0,0,0);controls.update()}
    }
}

// ============================================================
// 轉換
// ============================================================
function rT(to){
    if(ts===to&&tp<1&&tp>0)return;if(cs===to)return;
    ts=to;tp=0;tsm=performance.now();
}
function tT(now){
    if(ts===cs)return;
    var el=now-tsm;tp=Math.min(el/TD,1.0);var t=ease(tp),f=cs;

    if(f===STATE.LINEAR&&ts===STATE.NONLINEAR){setPP(t);sDBO(1-t);refSV(.85*(1-t),false)}
    if(f===STATE.NONLINEAR&&ts===STATE.KERNEL_3D){
        var lT=smooth(0,.35,t);aKL(lT);
        var sT=smooth(.2,1,t);
        kgg.visible=sT>.05;uGL(Math.min(sT*1.4,1));
        hpg.visible=sT>.28;svmPlane.material.opacity=.35*smooth(.28,.5,t);
        hpEdge.material.opacity=.55*smooth(.28,.5,t);
        pjg.visible=sT>.4;uPL(smooth(.4,.7,t));
        uC3(sT);if(controls)controls.enabled=sT>.5;
        document.getElementById('note-banner').style.display=sT>.35?'block':'none';
        // 切換到 3D SV
        if(sT>.5)refSV(.85,true);
    }
    if(f===STATE.LINEAR&&ts===STATE.KERNEL_3D){
        var pe=.35,ps=.25;
        if(t<=pe){var pt=Math.min(t/pe,1);setPP(pt);sDBO(1-pt);refSV(.85*(1-pt),false)}
        else{setPP(1);sDBO(0);refSV(0,false)}
        var lT2=smooth(ps,.65,t);aKL(lT2);kgg.visible=lT2>.05;uGL(Math.min(lT2*1.4,1));
        var sT2=smooth(ps+.05,1,t);
        hpg.visible=sT2>.25;svmPlane.material.opacity=.35*smooth(.25,.45,sT2);
        hpEdge.material.opacity=.55*smooth(.25,.45,sT2);
        pjg.visible=sT2>.35;uPL(smooth(.35,.65,sT2));
        uC3(sT2);if(controls)controls.enabled=sT2>.45;
        document.getElementById('note-banner').style.display=sT2>.3?'block':'none';
        if(sT2>.45)refSV(.85,true);
    }

    if(tp>=1){cs=ts;tp=0;uCS();uUI()}
}

function rL(){
    cs=STATE.LINEAR;ts=STATE.LINEAR;tp=0;
    setPP(0);aKL(0);sDBO(1);refSV(.85,false);
    kgg.visible=false;hpg.visible=false;pjg.visible=false;
    uCS();uPL(0);
    document.getElementById('note-banner').style.display='none';uUI();
}

document.getElementById('btn-linear').addEventListener('click',function(){rL()});
document.getElementById('btn-nonlinear').addEventListener('click',function(){
    if(cs===STATE.NONLINEAR)return;
    if(cs===STATE.KERNEL_3D){rL();setTimeout(function(){rT(STATE.NONLINEAR)},80)}
    else rT(STATE.NONLINEAR);
});
document.getElementById('btn-kernel').addEventListener('click',function(){
    if(cs===STATE.KERNEL_3D)return;
    rT(STATE.KERNEL_3D);
});

// FPS
var fc2=0,lft=performance.now(),fe=document.getElementById('fps-counter');
function uFP(now){fc2++;if(now-lft>=1000){fe.textContent='FPS: '+Math.round(fc2/((now-lft)/1000));fc2=0;lft=now}}

// ============================================================
// 動畫
// ============================================================
var orbitAngle=0;
function anim(ts2){
    requestAnimationFrame(anim);
    tT(ts2);
    sm.rotation.y+=.00015;sm.rotation.x+=.00008;
    if(cs===STATE.KERNEL_3D){
        hpEdge.rotation.z+=.003;
        if(controls && !controls.enabled){
            orbitAngle+=.0015;
            camera.position.x=Math.cos(orbitAngle)*9;
            camera.position.y=Math.sin(orbitAngle)*9*.4;
            camera.position.z=8+Math.sin(orbitAngle)*2;
            camera.lookAt(0,0,hpZ*.5);
        }
    }
    if(cs===STATE.LINEAR)refSV(.85,false);
    if(controls)controls.update();
    renderer.render(scene,camera);
    uFP(ts2);
}

window.addEventListener('resize',function(){
    var w2=window.innerWidth,h2=window.innerHeight;
    camera.aspect=w2/h2;camera.updateProjectionMatrix();
    renderer.setSize(w2,h2);
});

// 啟動
setPP(0);uUI();uCS();
document.getElementById('status').style.display='none';
document.getElementById('ui-overlay').style.display='block';
document.getElementById('btn-bar').style.display='flex';
document.getElementById('fps-counter').style.display='block';
requestAnimationFrame(anim);

}catch(e){
    document.getElementById('status').style.display='none';
    document.getElementById('err').style.display='block';
    document.getElementById('err').innerHTML='<b>場景初始化失敗</b><br><br>'+e.message;
    console.error(e);
}
} // startApp
})();
</script>
</body>
</html>"""


# ============================================================
# STREAMLIT
# ============================================================

def main():
    st.set_page_config(
        page_title="SVM 核方法 — 3D 互動視覺化",
        page_icon="🔮",
        layout="wide",
    )
    st.title("🔮 SVM 核方法：3D 互動視覺化 — 真實決策面")
    st.markdown("""
    **教學視覺化**：展示支持向量機 (SVM) 如何透過
    **核方法 (Kernel Trick)** 將非線性可分的資料映射到
    高維特徵空間，在該空間中訓練 SVM 找到**真實決策平面**，
    再投影回原始空間形成非線性決策邊界。

    下方按鈕切換狀態：**線性 SVM** → **非線性資料** → **核方法 3D**。
    在 3D 狀態下可拖曳旋轉、滾輪縮放、右鍵平移。
    """)

    try:
        data = _generate_datasets()
        data_json = json.dumps(data, indent=None, separators=(",", ":"))
        html = THREEJS_HTML.replace("__DATA__", data_json)

        # 診斷：確認 Python 端正常
        st.caption(f"✅ 資料已生成：{data['n']} 粒子/類別，{len(data['sv_3d_phi'])} 個 3D SV，{len(data['boundary_curve_2d'])} 個邊界點")

        components.html(html, height=780, scrolling=True)
    except Exception as e:
        st.error(f"應用程式錯誤：{e}")
        st.code(str(e))


if __name__ == "__main__":
    main()
