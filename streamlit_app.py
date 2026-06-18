"""
streamlit_app.py — SVM 核方法 3D 互動視覺化
============================================
完全自包含版本：所有程式碼內聯，不依賴外部 src/ 模組。
執行: streamlit run streamlit_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json

# ============================================================
# 內聯資料生成（不依賴 src/）
# ============================================================

def _generate_datasets(n_per_class=80):
    rng = np.random.RandomState(42)

    class_a_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([-3, -3])
    class_b_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([3, 3])
    X_linear = np.vstack([class_a_lin, class_b_lin])
    y_linear = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    class_a_nonlin = rng.randn(n_per_class, 2) * 1.0
    angles = rng.uniform(0, 2*np.pi, n_per_class)
    radii = rng.uniform(3.5, 5.0, n_per_class)
    class_b_nonlin = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

    try:
        from sklearn.svm import SVC
        svm = SVC(kernel="linear", C=1e10, random_state=42)
        svm.fit(X_linear, y_linear)
        w = svm.coef_[0].astype(np.float64)
        b = float(svm.intercept_[0])
        sv_indices = [int(i) for i in svm.support_]
    except ImportError:
        ca = class_a_lin.mean(axis=0); cb = class_b_lin.mean(axis=0)
        w = cb - ca; w = w / np.linalg.norm(w)
        b = float(-np.dot(w, (ca+cb)/2))
        da = np.abs(np.dot(class_a_lin, w)+b); db = np.abs(np.dot(class_b_lin, w)+b)
        sv_indices = (
            [int(i) for i in np.where(da <= np.percentile(da,20))[0]] +
            [int(n_per_class+i) for i in np.where(db <= np.percentile(db,20))[0]]
        )

    w_norm = w / np.linalg.norm(w)
    perp = np.array([-w_norm[1], w_norm[0]], dtype=np.float64)
    ext = 7.0; pc = -b*w_norm
    p1 = pc + perp*ext; p2 = pc - perp*ext
    p1p = p1 + w_norm; p2p = p2 + w_norm
    p1n = p1 - w_norm; p2n = p2 - w_norm

    def _kz(pts): return np.exp(-np.sum(pts**2, axis=1))
    zlr = _kz(class_a_lin); zlb = _kz(class_b_lin)
    znr = _kz(class_a_nonlin); znb = _kz(class_b_nonlin)
    sep_z = float(0.5*(np.mean(znr)+np.mean(znb)))

    return {
        "n": n_per_class,
        "linear_red": class_a_lin.tolist(), "linear_blue": class_b_lin.tolist(),
        "nonlinear_red": class_a_nonlin.tolist(), "nonlinear_blue": class_b_nonlin.tolist(),
        "z_linear_red": zlr.tolist(), "z_linear_blue": zlb.tolist(),
        "z_nonlinear_red": znr.tolist(), "z_nonlinear_blue": znb.tolist(),
        "w": w.tolist(), "b": b, "sv_indices": sv_indices,
        "db_line": [p1.tolist(), p2.tolist()],
        "margin_pos": [p1p.tolist(), p2p.tolist()],
        "margin_neg": [p1n.tolist(), p2n.tolist()],
        "hyperplane_z": sep_z,
    }


# ============================================================
# THREE.JS HTML (完全自包含)
# ============================================================

THREEJS_HTML_TEMPLATE = r"""<!DOCTYPE html>
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
    #status .dot:nth-child(2){animation-delay:0.2s}
    #status .dot:nth-child(3){animation-delay:0.4s}
    @keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
    .dot-g{background:#0f0}.dot-r{background:#f00}.dot-y{background:#ff0}
    #err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#f44;font-size:16px;z-index:100;text-align:center;display:none;background:rgba(0,0,0,0.85);padding:20px 30px;border-radius:10px;border:1px solid #f44;max-width:500px;line-height:1.6}
    #ui-overlay{position:absolute;top:24px;left:24px;background:rgba(0,0,0,0.75);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:22px 26px;color:#fff;max-width:420px;backdrop-filter:blur(8px);pointer-events:none;display:none}
    #state-name{font-size:28px;font-weight:bold;margin-bottom:6px;letter-spacing:2px}
    #formula{font-size:15px;color:#fc0;margin-bottom:12px;word-break:break-all;line-height:1.4}
    #explanation{font-size:14px;color:#bbb;line-height:1.7}
    #explanation .hl{color:#0f0}#explanation .wn{color:#ff0}
    #btn-bar{position:absolute;bottom:32px;left:50%;transform:translateX(-50%);display:none;gap:14px;z-index:20}
    #btn-bar button{padding:14px 26px;font-size:15px;font-weight:bold;letter-spacing:1px;cursor:pointer;border:2px solid rgba(0,255,255,0.6);background:rgba(0,20,40,0.85);color:#0ff;border-radius:10px;font-family:inherit;transition:all 0.25s;backdrop-filter:blur(6px);white-space:nowrap}
    #btn-bar button:hover{background:rgba(0,255,255,0.18);border-color:#0ff;box-shadow:0 0 24px rgba(0,255,255,0.45)}
    #btn-bar button.active{background:rgba(0,255,255,0.25);border-color:#fff;color:#fff}
    #note-banner{position:absolute;bottom:110px;left:50%;transform:translateX(-50%);color:#888;font-size:13px;font-family:inherit;text-align:center;display:none;background:rgba(0,0,0,0.6);padding:8px 20px;border-radius:6px;white-space:nowrap}
    #fps-counter{position:absolute;top:12px;right:16px;color:#444;font-size:11px;font-family:monospace;z-index:10;display:none}
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

<!-- Three.js + OrbitControls from jsdelivr CDN (global script, no importmap) -->
<script>
var __three_loaded = false;
var __orbit_loaded = false;
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"
    onload="__three_loaded=true;document.getElementById('status-msg').textContent='Three.js 已載入'"
    onerror="document.getElementById('err').style.display='block';document.getElementById('err').innerHTML='<b>Three.js 載入失敗</b><br>CDN 連線異常，請檢查網路或稍後再試'">
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/controls/OrbitControls.js"
    onload="__orbit_loaded=true;document.getElementById('status-msg').textContent='OrbitControls 已載入'"
    onerror="document.getElementById('status-msg').textContent='OrbitControls 載入失敗（將使用基本視角）'">
</script>

<script>
(function(){
'use strict';
var DATA = __DATA__;
var N = DATA.n;

// 等待所有腳本載入
function waitForScripts(cb) {
    var attempts = 0;
    var maxAttempts = 100;
    function check() {
        attempts++;
        if (typeof THREE !== 'undefined') {
            cb(true);
        } else if (attempts >= maxAttempts) {
            cb(false);
        } else {
            setTimeout(check, 100);
        }
    }
    check();
}

waitForScripts(function(ok) {
    if (!ok) {
        document.getElementById('status').style.display = 'none';
        var errEl = document.getElementById('err');
        errEl.style.display = 'block';
        errEl.innerHTML = '<b>Three.js 載入逾時</b><br><br>可能原因：<br>1. CDN 連線異常<br>2. 瀏覽器封鎖外部腳本<br>3. 網路防火牆限制<br><br>請嘗試重新整理頁面或更換網路環境';
        return;
    }
    startApp();
});

function startApp() {
try {

// ============================================================
// 狀態機
// ============================================================
var STATE = { LINEAR:0, NONLINEAR:1, KERNEL_3D:2 };
var cs = STATE.LINEAR, ts = STATE.LINEAR;
var tp = 0, tsm = 0;
var TD = 2800;

function ease(t){return t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2}
function lerp(a,b,t){return a+(b-a)*t}
function smooth(e0,e1,x){var t2=Math.max(0,Math.min((x-e0)/(e1-e0),1));return t2*t2*(3-2*t2)}

// ============================================================
// 粒子
// ============================================================
var RP=[], BP=[];
for(var i=0;i<N;i++){
    var lr=DATA.linear_red[i], nr=DATA.nonlinear_red[i];
    RP.push({lp:new THREE.Vector3(lr[0],lr[1],0),np:new THREE.Vector3(nr[0],nr[1],0),kz:DATA.z_nonlinear_red[i],cp:new THREE.Vector3(lr[0],lr[1],0),sv:DATA.sv_indices.indexOf(i)!==-1});
}
for(var i=0;i<N;i++){
    var lb=DATA.linear_blue[i], nb=DATA.nonlinear_blue[i];
    BP.push({lp:new THREE.Vector3(lb[0],lb[1],0),np:new THREE.Vector3(nb[0],nb[1],0),kz:DATA.z_nonlinear_blue[i],cp:new THREE.Vector3(lb[0],lb[1],0),sv:DATA.sv_indices.indexOf(N+i)!==-1});
}

// ============================================================
// 場景
// ============================================================
var W=window.innerWidth, H=window.innerHeight;
var scene=new THREE.Scene();
var camera=new THREE.PerspectiveCamera(50,W/H,0.1,100);
camera.position.set(0,0,14);camera.lookAt(0,0,0);
var renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W,H);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setClearColor(0x050510);
document.body.appendChild(renderer.domElement);

var controls=null;
if(typeof THREE.OrbitControls!=='undefined'){
    controls=new THREE.OrbitControls(camera,renderer.domElement);
    controls.enabled=false;controls.enableDamping=true;controls.dampingFactor=0.08;
    controls.minDistance=4;controls.maxDistance=30;controls.maxPolarAngle=Math.PI*0.75;
}

// 燈光
scene.add(new THREE.AmbientLight(0x222244,0.4));
var dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(0,8,12);scene.add(dl);

// ----- 星空 -----
var sg=new THREE.BufferGeometry();
var sc=1800, sa=new Float32Array(sc*3);
for(var i=0;i<sc*3;i+=3){sa[i]=(Math.random()-0.5)*45;sa[i+1]=(Math.random()-0.5)*45;sa[i+2]=(Math.random()-0.5)*22-11}
sg.setAttribute('position',new THREE.BufferAttribute(sa,3));
var sm=new THREE.Points(sg,new THREE.PointsMaterial({color:0xccccff,size:0.035,transparent:true,opacity:0.75,blending:THREE.AdditiveBlending,depthWrite:false}));
scene.add(sm);

// 星雲
var ng=new THREE.BufferGeometry(), nc2=400, np2=new Float32Array(nc2*3), ncl=new Float32Array(nc2*3);
for(var i=0;i<nc2*3;i+=3){np2[i]=(Math.random()-0.5)*24;np2[i+1]=(Math.random()-0.5)*24;np2[i+2]=(Math.random()-0.5)*12-6;ncl[i]=0.08+Math.random()*0.12;ncl[i+1]=0.02+Math.random()*0.06;ncl[i+2]=0.15+Math.random()*0.25}
ng.setAttribute('position',new THREE.BufferAttribute(np2,3));ng.setAttribute('color',new THREE.BufferAttribute(ncl,3));
scene.add(new THREE.Points(ng,new THREE.PointsMaterial({size:0.55,vertexColors:true,transparent:true,opacity:0.25,blending:THREE.AdditiveBlending,depthWrite:false})));

// ----- 資料粒子 -----
var rg=new THREE.BufferGeometry(), ra=new Float32Array(N*3);
rg.setAttribute('position',new THREE.BufferAttribute(ra,3));
var rp2=new THREE.Points(rg,new THREE.PointsMaterial({color:0xff2233,size:0.20,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));
scene.add(rp2);

var bg=new THREE.BufferGeometry(), ba=new Float32Array(N*3);
bg.setAttribute('position',new THREE.BufferAttribute(ba,3));
var bp2=new THREE.Points(bg,new THREE.PointsMaterial({color:0x2266ff,size:0.20,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));
scene.add(bp2);

// ----- 輔助函式 -----
function lm(pA,pB,color,width,opacity){
    var dir=new THREE.Vector3().subVectors(pB,pA), len=dir.length();
    var mid=new THREE.Vector3().addVectors(pA,pB).multiplyScalar(0.5);
    var geom=new THREE.CylinderGeometry(width,width,len,6,1);
    var mat=new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:opacity,depthTest:true});
    var mesh=new THREE.Mesh(geom,mat);mesh.position.copy(mid);
    mesh.setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize()));
    return mesh;
}

// ----- 決策邊界 -----
var dbg=new THREE.Group();scene.add(dbg);
(function(){
    var p1=new THREE.Vector3(DATA.db_line[0][0],DATA.db_line[0][1],0.02);
    var p2=new THREE.Vector3(DATA.db_line[1][0],DATA.db_line[1][1],0.02);
    var mp1=new THREE.Vector3(DATA.margin_pos[0][0],DATA.margin_pos[0][1],0.01);
    var mp2=new THREE.Vector3(DATA.margin_pos[1][0],DATA.margin_pos[1][1],0.01);
    var mn1=new THREE.Vector3(DATA.margin_neg[0][0],DATA.margin_neg[0][1],0.01);
    var mn2=new THREE.Vector3(DATA.margin_neg[1][0],DATA.margin_neg[1][1],0.01);
    dbg.add(lm(p1,p2,0x00ff88,0.08,1.0));
    dbg.add(lm(mp1,mp2,0xffcc00,0.04,0.9));
    dbg.add(lm(mn1,mn2,0xffcc00,0.04,0.9));
})();

// ----- SV 指標 -----
var svg=new THREE.Group();scene.add(svg);
var svrg=new THREE.TorusGeometry(0.32,0.05,8,20);
function refSV(op){
    while(svg.children.length>0)svg.remove(svg.children[0]);
    var all=RP.concat(BP);
    for(var i=0;i<all.length;i++){
        if(!all[i].sv)continue;
        var ring=new THREE.Mesh(svrg,new THREE.MeshBasicMaterial({color:0xffff44,transparent:true,opacity:op,depthTest:true}));
        ring.position.copy(all[i].cp);ring.position.z+=0.03;svg.add(ring);
    }
}
refSV(0.85);

// ----- 核網格 -----
var kgg=new THREE.Group();kgg.visible=false;scene.add(kgg);
var GR=28, GE=7.5, gn=[];
var gdg=new THREE.SphereGeometry(0.045,4,4);
for(var i=0;i<=GR;i++)for(var j=0;j<=GR;j++){
    var fx=(i/GR-0.5)*2*GE, fy=(j/GR-0.5)*2*GE;
    var gm2=new THREE.Mesh(gdg,new THREE.MeshBasicMaterial({color:0x8888ff,transparent:true,opacity:0.55,depthTest:true}));
    gm2.position.set(fx,fy,0);gm2.userData={bx:fx,by:fy};kgg.add(gm2);gn.push(gm2);
}

// ----- 超平面 -----
var hpg=new THREE.Group();hpg.visible=false;scene.add(hpg);
var hpM=new THREE.MeshBasicMaterial({color:0x00ffff,side:THREE.DoubleSide,transparent:true,opacity:0.28,depthWrite:false});
var hp=new THREE.Mesh(new THREE.PlaneGeometry(10,10),hpM);hp.position.z=DATA.hyperplane_z;hpg.add(hp);
var hrM=new THREE.MeshBasicMaterial({color:0x00ffff,transparent:true,opacity:0.5,depthTest:true});
var hr=new THREE.Mesh(new THREE.TorusGeometry(4.2,0.06,12,80),hrM);hr.position.z=DATA.hyperplane_z;hpg.add(hr);

// ----- 投影 -----
var pjg=new THREE.Group();pjg.visible=false;scene.add(pjg);
var pcM=new THREE.MeshBasicMaterial({color:0x00ff88,transparent:true,opacity:0.75,depthTest:true});
var pc2=new THREE.Mesh(new THREE.TorusGeometry(2.0,0.05,16,80),pcM);pjg.add(pc2);
var plg=new THREE.Group();pjg.add(plg);

// 座標軸
var ag=new THREE.Group();
ag.add(lm(new THREE.Vector3(-8,0,0),new THREE.Vector3(8,0,0),0x444444,0.02,0.25));
ag.add(lm(new THREE.Vector3(0,-8,0),new THREE.Vector3(0,8,0),0x444444,0.02,0.25));
ag.add(lm(new THREE.Vector3(0,0,-3),new THREE.Vector3(0,0,3),0x444444,0.02,0.25));
scene.add(ag);

// ============================================================
// 位置更新
// ============================================================
function setPP(mt){
    var ra2=rg.attributes.position.array, ba2=bg.attributes.position.array;
    for(var i=0;i<N;i++){
        var r=RP[i], b=BP[i];
        var rx=lerp(r.lp.x,r.np.x,mt), ry=lerp(r.lp.y,r.np.y,mt);
        var bx=lerp(b.lp.x,b.np.x,mt), by=lerp(b.lp.y,b.np.y,mt);
        r.cp.set(rx,ry,r.cp.z);b.cp.set(bx,by,b.cp.z);
        var ri=i*3,bi=i*3;
        ra2[ri]=rx;ra2[ri+1]=ry;ra2[ri+2]=r.cp.z;
        ba2[bi]=bx;ba2[bi+1]=by;ba2[bi+2]=b.cp.z;
    }
    rg.attributes.position.needsUpdate=true;bg.attributes.position.needsUpdate=true;
}
function aKL(kt){
    var ra2=rg.attributes.position.array, ba2=bg.attributes.position.array;
    for(var i=0;i<N;i++){
        var rz=lerp(0,RP[i].kz,kt), bz=lerp(0,BP[i].kz,kt);
        RP[i].cp.z=rz;BP[i].cp.z=bz;
        ra2[i*3+2]=rz;ba2[i*3+2]=bz;
    }
    rg.attributes.position.needsUpdate=true;bg.attributes.position.needsUpdate=true;
}
function uGL(kt){for(var i=0;i<gn.length;i++){var n=gn[i],x=n.userData.bx,y=n.userData.by;n.position.z=Math.exp(-(x*x+y*y))*kt;n.material.opacity=0.2+kt*0.35}}
function uC3(rt){camera.position.set(rt*5.5,rt*5.5,14-rt*5);camera.lookAt(0,0,lerp(0,DATA.hyperplane_z,rt))}
function uPL(op){
    while(plg.children.length>0)plg.remove(plg.children[0]);
    if(op<=0)return;var all=RP.concat(BP);
    for(var i=0;i<all.length;i+=4){var p=all[i],t=p.cp.clone(),b2=p.cp.clone();b2.z=0;plg.add(lm(t,b2,0x3355ff,0.015,op*0.35))}
}

function sDBO(o){for(var i=0;i<dbg.children.length;i++){if(dbg.children[i].material)dbg.children[i].material.opacity=o*0.9}}

// ============================================================
// UI
// ============================================================
var SU=[
    {nm:'線性 SVM',cl:'#0ff',fm:'f(x) = w<sup>T</sup>x + b',hx:'資料是 <span class="hl">完美線性可分</span> 的。<br>SVM 找到 <span class="hl">最佳超平面</span>，<br>最大化兩個類別之間的邊界距離。<br><br><span style="color:#0f0">▬</span> 決策邊界：w<sup>T</sup>x + b = 0<br><span class="wn">▬</span> 邊界線：w<sup>T</sup>x + b = ±1<br><span class="wn">◯</span> 支持向量（最近的點）'},
    {nm:'非線性資料',cl:'#f0f',fm:'在 ℝ² 空間中不存在線性分隔器',hx:'資料 <span class="wn">無法</span> 用一條直線分開。<br>紅色集中在中心，藍色在外圍環繞。<br>沒有任何直線可以將它們分離。<br><br><span class="wn">？</span> SVM 該如何處理？<br><span style="color:#888;">→</span> <b>核方法</b> 將資料映射到<br>更高維度的空間，使其<br><span class="hl">變得線性可分</span>！'},
    {nm:'核方法 3D',cl:'#ff0',fm:'Φ(x₁,x₂) = (x₁, x₂, e<sup>−(x₁²+x₂²)</sup>)',hx:'資料經由核函數 <span class="hl">提升到 3D</span>！<br>K(x,y) = exp(−γ||x−y||²)<br><br><span style="color:#0ff">▭</span> 3D 空間中的分離超平面<br><span style="color:#0f0">○</span> 投影回 2D 的圓形邊界<br><br><span class="hl">在高維空間中，資料<br>變得線性可分！</span>'}
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
    if(cs===STATE.KERNEL_3D){uC3(1);if(controls)controls.enabled=true}
    else{camera.position.set(0,0,14);camera.lookAt(0,0,0);if(controls){controls.enabled=false;controls.target.set(0,0,0);controls.update()}}
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

    if(f===STATE.LINEAR&&ts===STATE.NONLINEAR){setPP(t);sDBO(1-t);refSV(0.85*(1-t))}

    if(f===STATE.NONLINEAR&&ts===STATE.KERNEL_3D){
        var lT=smooth(0.0,0.35,t);aKL(lT);
        var sT=smooth(0.2,1.0,t);
        kgg.visible=sT>0.05;uGL(Math.min(sT*1.4,1.0));
        hpg.visible=sT>0.25;hpM.opacity=0.28*smooth(0.25,0.5,t);hrM.opacity=0.5*smooth(0.25,0.5,t);
        pjg.visible=sT>0.35;pcM.opacity=0.75*smooth(0.35,0.6,t);uPL(smooth(0.35,0.7,t));
        uC3(sT);if(controls)controls.enabled=sT>0.5;
        document.getElementById('note-banner').style.display=sT>0.4?'block':'none';
    }

    if(f===STATE.LINEAR&&ts===STATE.KERNEL_3D){
        var pe=0.35,ps=0.25;
        if(t<=pe){var pt=Math.min(t/pe,1.0);setPP(pt);sDBO(1-pt);refSV(0.85*(1-pt))}
        else{setPP(1);sDBO(0);refSV(0)}
        var lT2=smooth(ps,0.65,t);aKL(lT2);kgg.visible=lT2>0.05;uGL(Math.min(lT2*1.4,1.0));
        var sT2=smooth(ps+0.05,1.0,t);
        hpg.visible=sT2>0.2;hpM.opacity=0.28*smooth(0.2,0.45,sT2);hrM.opacity=0.5*smooth(0.2,0.45,sT2);
        pjg.visible=sT2>0.3;pcM.opacity=0.75*smooth(0.3,0.55,sT2);uPL(smooth(0.3,0.65,sT2));
        uC3(sT2);if(controls)controls.enabled=sT2>0.45;
        document.getElementById('note-banner').style.display=sT2>0.35?'block':'none';
    }

    if(tp>=1.0){cs=ts;tp=0;uCS();uUI()}
}

function rL(){
    cs=STATE.LINEAR;ts=STATE.LINEAR;tp=0;
    setPP(0);aKL(0);sDBO(1);refSV(0.85);
    kgg.visible=false;hpg.visible=false;pjg.visible=false;
    uCS();uPL(0);document.getElementById('note-banner').style.display='none';uUI();
}

// ============================================================
// 按鈕
// ============================================================
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
// 動畫迴圈
// ============================================================
function anim(ts2){
    requestAnimationFrame(anim);
    tT(ts2);
    sm.rotation.y+=0.00015;sm.rotation.x+=0.00008;
    if(cs===STATE.KERNEL_3D){hr.rotation.z+=0.003;pc2.rotation.z+=0.002}
    refSV(cs===STATE.LINEAR?0.85:0);
    if(controls)controls.update();
    renderer.render(scene,camera);
    uFP(ts2);
}

window.addEventListener('resize',function(){
    var w2=window.innerWidth,h2=window.innerHeight;
    camera.aspect=w2/h2;camera.updateProjectionMatrix();
    renderer.setSize(w2,h2);
});

// ============================================================
// 啟動
// ============================================================
setPP(0);uUI();uCS();
document.getElementById('status').style.display='none';
document.getElementById('ui-overlay').style.display='block';
document.getElementById('btn-bar').style.display='flex';
document.getElementById('fps-counter').style.display='block';
requestAnimationFrame(anim);

}catch(e){
    document.getElementById('status').style.display='none';
    var errEl2=document.getElementById('err');
    errEl2.style.display='block';
    errEl2.innerHTML='<b>場景初始化失敗</b><br><br>錯誤：'+e.message+'<br><br>請嘗試重新整理頁面';
    console.error(e);
}
} // end startApp

})(); // IIFE
</script>
</body>
</html>"""


# ============================================================
# STREAMLIT 進入點
# ============================================================

def main():
    st.set_page_config(
        page_title="SVM 核方法 — 3D 互動視覺化",
        page_icon="🔮",
        layout="wide",
    )

    st.title("🔮 SVM 核方法：3D 互動視覺化")
    st.markdown("""
    **教學視覺化**：展示支持向量機 (SVM) 如何透過
    **核方法 (Kernel Trick)** 將非線性可分的資料映射到
    高維空間中，使其變得線性可分。

    使用下方 3D 畫面中的按鈕逐步切換狀態：
    **線性 SVM** → **非線性資料** → **核方法 3D**。
    在 3D 狀態下，可拖曳旋轉、滾輪縮放、右鍵平移。
    """)

    try:
        data = _generate_datasets()
        data_json = json.dumps(data, indent=None, separators=(",", ":"))
        html = THREEJS_HTML_TEMPLATE.replace("__DATA__", data_json)
        components.html(html, height=750, scrolling=True)
    except Exception as e:
        st.error(f"應用程式錯誤：{e}")
        st.code(str(e))


if __name__ == "__main__":
    main()
