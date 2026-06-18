"""
streamlit_app.py — SVM Kernel Trick 3D (Three.js smooth 60fps animation)

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json

DEFAULT_Z = 8.0; DEFAULT_N = 80

def _gen(n=80, zs=8.0):
    rng=np.random.RandomState(42)
    al=rng.randn(n,2)*1.2+[-3,-3]; bl=rng.randn(n,2)*1.2+[3,3]
    X=np.vstack([al,bl]); y=np.hstack([np.zeros(n),np.ones(n)])
    an=rng.randn(n,2)*1.0
    ang=rng.uniform(0,2*np.pi,n); rad=rng.uniform(3.5,5,n)
    bn=np.column_stack([rad*np.cos(ang),rad*np.sin(ang)])
    try:
        from sklearn.svm import SVC
        s=SVC(kernel="linear",C=1e10,random_state=42); s.fit(X,y)
        w=s.coef_[0].astype(np.float64); b2=float(s.intercept_[0]); sv2=[int(i) for i in s.support_]
    except:
        ca=al.mean(0); cb=bl.mean(0); w=cb-ca; w/=np.linalg.norm(w); b2=float(-np.dot(w,(ca+cb)/2))
        sv2=[int(i) for i in np.where(np.abs(np.dot(al,w)+b2)<=np.percentile(np.abs(np.dot(al,w)+b2),20))[0]]+[int(n+i) for i in np.where(np.abs(np.dot(bl,w)+b2)<=np.percentile(np.abs(np.dot(bl,w)+b2),20))[0]]
    wn=w/np.linalg.norm(w); perp=np.array([-wn[1],wn[0]],dtype=np.float64); pc=-b2*wn; e=7.0
    p1=pc+perp*e; p2=pc-perp*e

    def kz(p): return zs*np.exp(-np.sum(p**2,axis=1))
    def phi(p): r2=np.sum(p**2,axis=1); return np.column_stack([p,zs*np.exp(-r2)])
    pa=phi(an); pb=phi(bn); znr=kz(an); znb=kz(bn)
    rc=pa.mean(0); bc=pb.mean(0); w3=rc-bc; w3/=np.linalg.norm(w3); mid=(rc+bc)/2; b3=float(-np.dot(w3,mid))
    da=np.abs(np.dot(pa,w3)+b3); db2=np.abs(np.dot(pb,w3)+b3)
    sva=np.where(da<=np.percentile(da,10))[0]; svb=np.where(db2<=np.percentile(db2,10))[0]
    sv3=np.vstack([pa[sva],pb[svb]]).tolist()

    cv=[]
    for i in range(120):
        th=2*np.pi*i/120; ct,st=np.cos(th),np.sin(th); lo,hi=0.,7.
        fl=w3[0]*lo*ct+w3[1]*lo*st+w3[2]*zs*np.exp(-lo*lo)+b3
        fh=w3[0]*hi*ct+w3[1]*hi*st+w3[2]*zs*np.exp(-hi*hi)+b3
        if fl*fh>0: continue
        for _ in range(30):
            mid=(lo+hi)/2; fm=w3[0]*mid*ct+w3[1]*mid*st+w3[2]*zs*np.exp(-mid*mid)+b3
            if abs(fm)<0.005: break
            if fl*fm<0: hi=mid;fh=fm
            else: lo=mid;fl=fm
        r=(lo+hi)/2; cv.append([float(r*ct),float(r*st)])

    return dict(n=n,al=al.tolist(),bl=bl.tolist(),an=an.tolist(),bn=bn.tolist(),
                znr=znr.tolist(),znb=znb.tolist(),w=w.tolist(),b2=b2,sv2=sv2,
                db1=p1.tolist(),db2=p2.tolist(),w3=w3.tolist(),b3=b3,sv3=sv3,cv=cv,zs=zs)

# ============================================================
THREE_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SVM Kernel Trick 3D</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0A0A1A;overflow:hidden;font-family:Arial,sans-serif}
canvas{display:block}
#ui{position:absolute;top:20px;left:20px;background:rgba(0,0,0,0.7);color:#fff;padding:16px 20px;border-radius:10px;max-width:360px;pointer-events:none;z-index:10;border:1px solid rgba(255,255,255,0.1)}
#state{font-size:22px;font-weight:bold;margin-bottom:4px}
#formula{font-size:13px;color:#fc0;margin-bottom:8px;font-family:monospace}
#info{font-size:12px;color:#aaa;line-height:1.5}
#btns{position:absolute;bottom:24px;left:50%;transform:translateX(-50%);display:flex;gap:10px;z-index:10}
#btns button{padding:11px 22px;font-size:14px;border:1px solid rgba(0,255,255,0.5);background:rgba(0,20,40,0.8);color:#0ff;border-radius:8px;cursor:pointer;transition:all .2s}
#btns button:hover{background:rgba(0,255,255,0.15)}
#btns button.active{background:rgba(0,255,255,0.2);border-color:#fff;color:#fff}
#fps{position:absolute;top:8px;right:12px;color:#333;font-size:10px;font-family:monospace;z-index:10}
#load{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#0ff;font-size:16px;z-index:100}
</style></head><body>
<div id="load">Loading 3D scene...</div>
<div id="ui"><div id="state">SVM Kernel Trick</div><div id="formula">f(x) = wᵀx + b</div><div id="info">Initializing visualization...</div></div>
<div id="btns">
<button id="b0" class="active">Linear SVM</button>
<button id="b1">Nonlinear</button>
<button id="b2">Kernel 3D</button>
</div><div id="fps"></div>
<script>
var DATA = __DATA__;
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
 defer onload="init()" onerror="document.getElementById('load').textContent='CDN failed'">
</script>
<script>
var N,scene,camera,renderer,redPts,bluePts,redGeo,blueGeo;
var dbLine,marginP,marginN,svRings=[],kGrid=[],hpPlane,projCurve,projLines=[];
var state=0, targetState=0, transT=0, transStart=0, transActive=false;
var STATES=['LINEAR','NONLINEAR','KERNEL3'];
var T_DUR=2200;
var STATE_TEXT=[
 {n:'Linear SVM',c:'#0ff',f:'f(x)=w<sup>T</sup>x+b',i:'Data <span style=color:#0f0>linearly separable</span>. SVM finds optimal hyperplane.<br><span style=color:#0f0>&#x2501;</span> Boundary: w<sup>T</sup>x+b=0<br><span style=color:#ff0>&#x2501;</span> Margins: &plusmn;1<br><span style=color:#ff0>&#x25CB;</span> Support vectors'},
 {n:'Nonlinear Data',c:'#f0f',f:'No linear separator in &#x211D;<sup>2</sup>',i:'Data <span style=color:#ff0>not separable</span> in 2D.<br>Red cluster at center, blue ring around.<br><span style=color:#ff0>&#x2192;</span> <b>Kernel trick</b> lifts to 3D!'},
 {n:'Kernel 3D',c:'#ff0',f:'&#x3A6;(x,y)=(x,y,'+DATA.zs.toFixed(0)+'&middot;e<sup>&#x2212;(x&sup2;+y&sup2;)</sup>)',i:'Data <span style=color:#0f0>lifted to 3D</span>!<br>Plane separates classes in kernel space.<br>Green curve = 2D projection boundary.'}
];
function ease(t){return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2}
function lerp(a,b,t){return a+(b-a)*t}
function init(){
 if(typeof THREE==='undefined'){document.getElementById('load').textContent='THREE missing';return}
 N=DATA.n;
 var W=window.innerWidth,H=window.innerHeight;
 scene=new THREE.Scene();
 scene.fog=new THREE.FogExp2(0x0A0A1A,0.0006);
 camera=new THREE.PerspectiveCamera(45,W/H,0.5,100);
 camera.position.set(3,-2,13);camera.lookAt(0,0,0);
 renderer=new THREE.WebGLRenderer({antialias:true});
 renderer.setSize(W,H);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
 renderer.setClearColor(0x0A0A1A);
 document.body.appendChild(renderer.domElement);
 scene.add(new THREE.AmbientLight(0x333355,0.5));
 var dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(3,8,10);scene.add(dl);

 // stars
 var sg=new THREE.BufferGeometry(),sc=2000,sa=new Float32Array(sc*3);
 for(var i=0;i<sc*3;i+=3){sa[i]=(Math.random()-.5)*50;sa[i+1]=(Math.random()-.5)*50;sa[i+2]=(Math.random()-.5)*25-5}
 sg.setAttribute('position',new THREE.BufferAttribute(sa,3));
 scene.add(new THREE.Points(sg,new THREE.PointsMaterial({color:0xccccff,size:.04,transparent:true,opacity:.8,blending:THREE.AdditiveBlending,depthWrite:false})));

 // grid
 var gh=new THREE.GridHelper(16,24,0x222244,0x111122);scene.add(gh);

 // particles
 redGeo=new THREE.BufferGeometry();var ra=new Float32Array(N*3);
 redGeo.setAttribute('position',new THREE.BufferAttribute(ra,3));
 redPts=new THREE.Points(redGeo,new THREE.PointsMaterial({color:0x22d3ee,size:.22,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));
 scene.add(redPts);
 blueGeo=new THREE.BufferGeometry();var ba=new Float32Array(N*3);
 blueGeo.setAttribute('position',new THREE.BufferAttribute(ba,3));
 bluePts=new THREE.Points(blueGeo,new THREE.PointsMaterial({color:0xa855f7,size:.22,blending:THREE.AdditiveBlending,depthWrite:false,transparent:true}));
 scene.add(bluePts);

 // decision boundary
 dbLine=line(DATA.db1[0],DATA.db1[1],0.02, DATA.db2[0],DATA.db2[1],0.02, 0x00ff88,0.08);
 marginP=line(DATA.db1[0]+DATA.w[0]*.14,DATA.db1[1]+DATA.w[1]*.14,0.01, DATA.db2[0]+DATA.w[0]*.14,DATA.db2[1]+DATA.w[1]*.14,0.01, 0xffcc00,0.04);
 marginN=line(DATA.db1[0]-DATA.w[0]*.14,DATA.db1[1]-DATA.w[1]*.14,0.01, DATA.db2[0]-DATA.w[0]*.14,DATA.db2[1]-DATA.w[1]*.14,0.01, 0xffcc00,0.04);

 // kernel grid
 var kg=new THREE.Group();kg.visible=false;scene.add(kg);
 var gr=24,ge=7.5,gd=new THREE.SphereGeometry(.05,4,4);
 for(var i=0;i<=gr;i++)for(var j=0;j<=gr;j++){
  var fx=(i/gr-.5)*2*ge,fy=(j/gr-.5)*2*ge;
  var m=new THREE.Mesh(gd,new THREE.MeshBasicMaterial({color:0x8888ff,transparent:true,opacity:.5,depthTest:true}));
  m.position.set(fx,fy,0);m.userData={bx:fx,by:fy};kg.add(m);kGrid.push(m);
 }

 // hyperplane
 var hpg=new THREE.Group();hpg.visible=false;scene.add(hpg);
 var hpZ=DATA.b3!==undefined? -DATA.b3/DATA.w3[2] : DATA.zs*0.35;
 var hpMat=new THREE.MeshBasicMaterial({color:0x00ffff,side:THREE.DoubleSide,transparent:true,opacity:.3,depthWrite:false});
 hpPlane=new THREE.Mesh(new THREE.PlaneGeometry(12,12),hpMat);hpPlane.position.z=hpZ;
 var q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(DATA.w3[0],DATA.w3[1],DATA.w3[2]).normalize());
 hpPlane.setRotationFromQuaternion(q);
 var n2=DATA.w3[0]*DATA.w3[0]+DATA.w3[1]*DATA.w3[1]+DATA.w3[2]*DATA.w3[2];
 var t2=-DATA.b3/n2;hpPlane.position.set(DATA.w3[0]*t2,DATA.w3[1]*t2,DATA.w3[2]*t2);
 hpg.add(hpPlane);

 // projection curve
 var pcg=new THREE.Group();pcg.visible=false;scene.add(pcg);
 if(DATA.cv&&DATA.cv.length>1){
  for(var ci=0;ci<DATA.cv.length;ci++){
   var c0=DATA.cv[ci],c1=DATA.cv[(ci+1)%DATA.cv.length];
   pcg.add(line(c0[0],c0[1],0.03,c1[0],c1[1],0.03,0x4ade80,0.05));
  }
 }

 // z pillar
 scene.add(line(0,0,-3,0,0,hpZ+4,0x335566,0.03));

  setPositions(0);updateUI();updateSV();
  document.getElementById('load').style.display='none';
  animate(0);
}
function line(x1,y1,z1,x2,y2,z2,color,width){
 var d=new THREE.Vector3(x2-x1,y2-y1,z2-z1),len=d.length();
 var mid=new THREE.Vector3((x1+x2)/2,(y1+y2)/2,(z1+z2)/2);
 var g=new THREE.CylinderGeometry(width,width,len,6,1);
 var m=new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:1,depthTest:true});
 var mesh=new THREE.Mesh(g,m);mesh.position.copy(mid);
 mesh.setRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize()));
 scene.add(mesh);return mesh;
}
function setPositions(mt){
 var ra=redGeo.attributes.position.array,ba=blueGeo.attributes.position.array;
 for(var i=0;i<N;i++){
  var rx=lerp(DATA.al[i][0],DATA.an[i][0],mt),ry=lerp(DATA.al[i][1],DATA.an[i][1],mt);
  var bx=lerp(DATA.bl[i][0],DATA.bn[i][0],mt),by=lerp(DATA.bl[i][1],DATA.bn[i][1],mt);
  ra[i*3]=rx;ra[i*3+1]=ry;ba[i*3]=bx;ba[i*3+1]=by;
 }
 redGeo.attributes.position.needsUpdate=true;blueGeo.attributes.position.needsUpdate=true;
}
function liftZ(kt){
 var ra=redGeo.attributes.position.array,ba=blueGeo.attributes.position.array;
 for(var i=0;i<N;i++){ra[i*3+2]=lerp(0,DATA.znr[i],kt);ba[i*3+2]=lerp(0,DATA.znb[i],kt)}
 redGeo.attributes.position.needsUpdate=true;blueGeo.attributes.position.needsUpdate=true;
}
function updateGrid(kt){for(var i=0;i<kGrid.length;i++){var n=kGrid[i],x=n.userData.bx,y=n.userData.by;n.position.z=Math.exp(-(x*x+y*y))*DATA.zs*kt}}
function setCam(rt){camera.position.set(rt*7,rt*3,13-rt*6);camera.lookAt(0,0,lerp(0,DATA.zs*.35,rt))}
function updateSV(){
 while(svRings.length>0){scene.remove(svRings.pop())}
 var rg=new THREE.TorusGeometry(.34,.055,8,20);
 var ra=redGeo.attributes.position.array,ba=blueGeo.attributes.position.array;
 DATA.sv2.forEach(function(idx){
  var arr=idx<N?ra:ba,i=idx<N?idx:idx-N;
  var r=new THREE.Mesh(rg,new THREE.MeshBasicMaterial({color:0xffbb24,transparent:true,opacity:.85,depthTest:true}));
  r.position.set(arr[i*3],arr[i*3+1],arr[i*3+2]+.05);scene.add(r);svRings.push(r);
 });
}
function updateUI(){
 var u=STATE_TEXT[state];
 document.getElementById('state').textContent=u.n;document.getElementById('state').style.color=u.c;
 document.getElementById('formula').innerHTML=u.f;document.getElementById('info').innerHTML=u.i;
 ['b0','b1','b2'].forEach(function(id,i){document.getElementById(id).className=i===state?'active':''});
}
function startTrans(to){targetState=to;transActive=true;transStart=performance.now();transT=0}
function tickTrans(now){
 if(!transActive)return;
 var el=now-transStart;transT=Math.min(el/T_DUR,1);var t=ease(transT),f=state;
 if(f===0&&targetState===1){setPositions(t);dbLine.material.opacity=1-t;marginP.material.opacity=(1-t)*.9;marginN.material.opacity=(1-t)*.9}
 if((f===1||f===0)&&targetState===2){
  if(f===0&&t<=.35){var pt=Math.min(t/.35,1);setPositions(pt);dbLine.material.opacity=1-pt;marginP.material.opacity=(1-pt)*.9;marginN.material.opacity=(1-pt)*.9}
  else if(f===0){setPositions(1);dbLine.material.opacity=0;marginP.material.opacity=0;marginN.material.opacity=0}
  var lt=smooth(f===0?.25:.0,f===0?.65:.35,t);liftZ(lt);
  var st=smooth(f===0?.3:.2,f===0?1:.85,t);
  kGrid[0].parent.visible=st>.05;updateGrid(Math.min(st*1.4,1));
  hpPlane.parent.visible=st>.25;hpPlane.material.opacity=.3*st;
  var pcg2=projCurve;if(!pcg2)pcg2=scene.children[scene.children.length-3];
  if(pcg2&&pcg2.type==='Group'){pcg2.visible=st>.35;pcg2.children.forEach(function(c){if(c.material)c.material.opacity=.85*st})}
  setCam(st);
 }
 if(transT>=1){state=targetState;transActive=false;setCam(state===2?1:0);if(state!==2){liftZ(0);setPositions(state===0?0:1);dbLine.material.opacity=state===0?1:0;marginP.material.opacity=state===0?.9:0;marginN.material.opacity=state===0?.9:0}updateUI()}
}
function smooth(e0,e1,x){var t2=Math.max(0,Math.min((x-e0)/(e1-e0),1));return t2*t2*(3-2*t2)}
document.getElementById('b0').onclick=function(){if(state===0)return;startTrans(2);setTimeout(function(){if(!transActive){state=0;setPositions(0);liftZ(0);dbLine.material.opacity=1;marginP.material.opacity=.9;marginN.material.opacity=.9;setCam(0);updateUI();updateSV()}},50)};
document.getElementById('b1').onclick=function(){if(state===1)return;if(state===2){state=0;setPositions(0);liftZ(0);dbLine.material.opacity=1;setCam(0);updateUI()}startTrans(1)};
document.getElementById('b2').onclick=function(){if(state===2)return;startTrans(2)};
var fc=0,lft=0;
function animate(ts){
 requestAnimationFrame(animate);
 tickTrans(ts);
 fc++;if(ts-lft>=1000){document.getElementById('fps').textContent='FPS:'+Math.round(fc/((ts-lft)/1000));fc=0;lft=ts}
 renderer.render(scene,camera);
}
window.addEventListener('resize',function(){camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight)});
</script></body></html>"""

def main():
    st.set_page_config(page_title="SVM Kernel 3D",page_icon="🔮",layout="wide")
    d=_gen();j=json.dumps(d)
    h=THREE_HTML.replace("__DATA__",j)
    components.html(h,height=750,scrolling=True)

if __name__=="__main__":main()
