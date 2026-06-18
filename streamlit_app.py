"""
streamlit_app.py — SVM Kernel Trick 3D Visualization

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

DEFAULT_Z_SCALE = 8.0
DEFAULT_N = 80
C_TEAL = '#22D3EE'; C_PURPLE = '#A855F7'; C_GOLD = '#FBBF24'
C_CURVE = '#4ADE80'; C_MARGIN = '#FDE68A'
C_GRID = 'rgba(100,116,139,0.12)'

def _lerp(a,b,t): return a+(b-a)*t

def _generate_datasets(n_per_class=DEFAULT_N, z_scale=DEFAULT_Z_SCALE):
    rng = np.random.RandomState(42)
    a_lin = rng.randn(n_per_class,2)*1.2 + [-3,-3]
    b_lin = rng.randn(n_per_class,2)*1.2 + [3,3]
    Xl = np.vstack([a_lin,b_lin]); yl = np.hstack([np.zeros(n_per_class),np.ones(n_per_class)])
    a_nl = rng.randn(n_per_class,2)*1.0
    ang = rng.uniform(0,2*np.pi,n_per_class); rad = rng.uniform(3.5,5,n_per_class)
    b_nl = np.column_stack([rad*np.cos(ang), rad*np.sin(ang)])
    try:
        from sklearn.svm import SVC
        s = SVC(kernel="linear",C=1e10,random_state=42); s.fit(Xl,yl)
        w = s.coef_[0].astype(np.float64); b2 = float(s.intercept_[0]); sv2 = [int(i) for i in s.support_]
    except ImportError:
        ca=a_lin.mean(0); cb=b_lin.mean(0); w=cb-ca; w/=np.linalg.norm(w); b2=float(-np.dot(w,(ca+cb)/2))
        da=np.abs(np.dot(a_lin,w)+b2); db=np.abs(np.dot(b_lin,w)+b2)
        sv2=[int(i) for i in np.where(da<=np.percentile(da,20))[0]]+[int(n_per_class+i) for i in np.where(db<=np.percentile(db,20))[0]]
    wn=w/np.linalg.norm(w); perp=np.array([-wn[1],wn[0]],dtype=np.float64); pc=-b2*wn; ext=7.0; p1=pc+perp*ext; p2=pc-perp*ext
    def _kz(pts): return z_scale*np.exp(-np.sum(pts**2,axis=1))
    def _phi(pts): r2=np.sum(pts**2,axis=1); return np.column_stack([pts,z_scale*np.exp(-r2)])
    pa=_phi(a_nl); pb=_phi(b_nl); znr=_kz(a_nl); znb=_kz(b_nl)
    rc=pa.mean(0); bc=pb.mean(0); w3=rc-bc; w3/=np.linalg.norm(w3); mid=(rc+bc)/2; b3=float(-np.dot(w3,mid))
    da3=np.abs(np.dot(pa,w3)+b3); db3=np.abs(np.dot(pb,w3)+b3)
    sva=np.where(da3<=np.percentile(da3,10))[0]; svb=np.where(db3<=np.percentile(db3,10))[0]; sv3p=np.vstack([pa[sva],pb[svb]])
    cv=_curve(w3[0],w3[1],w3[2],b3,z_scale); pX,pY,pZ=_plane(w3[0],w3[1],w3[2],b3,z_scale)
    return dict(n=n_per_class,z_scale=z_scale,a_lin=a_lin,b_lin=b_lin,a_nl=a_nl,b_nl=b_nl,
                znr=znr,znb=znb,w=w,b_2d=b2,sv_2d=sv2,p1=p1,p2=p2,
                w3=w3,b3=b3,sv_3d_phi=sv3p,curve_2d=cv,plane_pts=(pX,pY,pZ),red_c=rc,blue_c=bc)

def _curve(a,b,c,d,zs,n=150):
    pts=[]
    for i in range(n):
        th=2*np.pi*i/n; ct,st=np.cos(th),np.sin(th); lo,hi=0.,7.
        fl=a*lo*ct+b*lo*st+c*zs*np.exp(-lo*lo)+d; fh=a*hi*ct+b*hi*st+c*zs*np.exp(-hi*hi)+d
        if fl*fh>0: continue
        for _ in range(30):
            mid=(lo+hi)/2; fm=a*mid*ct+b*mid*st+c*zs*np.exp(-mid*mid)+d
            if abs(fm)<0.005: break
            if fl*fm<0: hi=mid;fh=fm
            else: lo=mid;fl=fm
        r=(lo+hi)/2; pts.append([float(r*ct),float(r*st)])
    return np.array(pts)

def _plane(a,b,c,d,zs,ext=7.0,res=40):
    xs=np.linspace(-ext,ext,res); ys=np.linspace(-ext,ext,res)
    X,Y=np.meshgrid(xs,ys); Z=-(a*X+b*Y+d)/c if abs(c)>1e-8 else np.zeros_like(X)
    return X,Y,np.clip(Z,-2,zs+4)

def get_data(n,z):
    ck=f"d{n}_{z}"
    if ck not in st.session_state: st.session_state[ck]=_generate_datasets(n,z)
    return st.session_state[ck]

# ============================================================
# Figures — t: 0→1 animation progress (via sidebar slider)
# ============================================================

def fig_linear(d, t=1.0):
    a,b=d["a_lin"],d["b_lin"]; sv=set(d["sv_2d"]); n=d["n"]; w,bb=d["w"],d["b_2d"]; wn=w/np.linalg.norm(w)
    ma=[i for i in range(n) if i not in sv]; mb=[i for i in range(n) if i+n not in sv]
    sa=[i for i in range(n) if i in sv]; sb=[i-n for i in sv if i>=n]
    perp=np.array([-wn[1],wn[0]]); pc=-bb*wn; ext=7.5
    xb=np.array([pc[0]-perp[0]*ext,pc[0]+perp[0]*ext]); yb=np.array([pc[1]-perp[1]*ext,pc[1]+perp[1]*ext])
    bd_op=np.clip(t/0.3,0,1); mg_op=np.clip((t-0.2)/0.4,0,1); sv_op=np.clip((t-0.5)/0.5,0,1); sv_sz=1+sv_op*13

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=a[ma,0],y=a[ma,1],mode='markers',name='類別 A',
        marker=dict(size=8,color=C_TEAL,opacity=0.85,line=dict(width=1,color='rgba(255,255,255,0.2)'))))
    fig.add_trace(go.Scatter(x=b[mb,0],y=b[mb,1],mode='markers',name='類別 B',
        marker=dict(size=8,color=C_PURPLE,opacity=0.85,line=dict(width=1,color='rgba(255,255,255,0.2)'))))
    fig.add_trace(go.Scatter(x=xb,y=yb,mode='lines',name='決策邊界 wᵀx+b=0',
        line=dict(color='#FFFFFF',width=2.5),opacity=bd_op))
    for s,lb in [(1,'+1'),(-1,'−1')]:
        o=s*wn; fig.add_trace(go.Scatter(x=xb+o[0],y=yb+o[1],mode='lines',
            name=f'邊界 {lb}',line=dict(color=C_MARGIN,width=1.5,dash='dash'),opacity=mg_op*0.85))
    fig.add_trace(go.Scatter(x=a[sa,0],y=a[sa,1],mode='markers',name='支持向量',
        marker=dict(size=sv_sz,color=C_GOLD,opacity=sv_op,line=dict(width=2.5,color='#FFF'),symbol='circle-open')))
    fig.add_trace(go.Scatter(x=b[sb,0],y=b[sb,1],mode='markers',showlegend=False,
        marker=dict(size=sv_sz,color=C_GOLD,opacity=sv_op,line=dict(width=2.5,color='#FFF'),symbol='circle-open')))
    fig.update_layout(template="plotly_dark",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁',gridcolor=C_GRID,zeroline=False,range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂',gridcolor=C_GRID,zeroline=False,range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
        margin=dict(l=40,r=20,t=30,b=40),hovermode='closest',height=600,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0,font=dict(color='#94A3B8'),bgcolor='rgba(0,0,0,0)'))
    return fig

def fig_nonlinear(d, t=1.0):
    a_lin,b_lin=d["a_lin"],d["b_lin"]; a_nl,b_nl=d["a_nl"],d["b_nl"]
    et=1-(1-t)**3
    ax=_lerp(a_lin[:,0],a_nl[:,0],et); ay=_lerp(a_lin[:,1],a_nl[:,1],et)
    bx=_lerp(b_lin[:,0],b_nl[:,0],et); by=_lerp(b_lin[:,1],b_nl[:,1],et)
    sz=9*(1-et)+6*et; ring_op=np.clip((t-0.7)/0.3,0,1)*0.7
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ax,y=ay,mode='markers',name='類別 A',
        marker=dict(size=sz,color=C_TEAL,opacity=0.85,line=dict(width=1,color='rgba(255,255,255,0.2)'))))
    fig.add_trace(go.Scatter(x=bx,y=by,mode='markers',name='類別 B',
        marker=dict(size=sz,color=C_PURPLE,opacity=0.85,line=dict(width=1,color='rgba(255,255,255,0.2)'))))
    tc=np.linspace(0,2*np.pi,200)
    fig.add_trace(go.Scatter(x=4.25*np.cos(tc),y=4.25*np.sin(tc),mode='lines',showlegend=False,
        line=dict(color='rgba(168,85,247,0.3)',width=1.5,dash='dot'),opacity=ring_op,hoverinfo='skip'))
    fig.update_layout(template="plotly_dark",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁',gridcolor=C_GRID,zeroline=False,range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂',gridcolor=C_GRID,zeroline=False,range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
        margin=dict(l=40,r=20,t=30,b=40),hovermode='closest',height=600,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0,font=dict(color='#94A3B8'),bgcolor='rgba(0,0,0,0)'))
    return fig

def fig_kernel3d(d, t=1.0):
    a,b=d["a_nl"],d["b_nl"]; znr,znb=d["znr"],d["znb"]; sv3=d["sv_3d_phi"]
    cv=d["curve_2d"]; Xp,Yp,Zp=d["plane_pts"]; n=d["n"]; zs=d["z_scale"]; w3,b3=d["w3"],d["b3"]
    zl=np.clip(t/0.5,0,1); st2=np.clip((t-0.4)/0.6,0,1); st_f=float(st2)
    za=znr*zl; zb=znb*zl; zsv=sv3[:,2]*zl if len(sv3)>0 else []
    pz=Zp*st2; po=0.5*st_f
    cc='rgba(200,220,255,%.2f)'%(0.35*st_f)
    sc0='rgba(34,211,238,%.2f)'%(0.55*st_f); sc1='rgba(168,85,247,%.2f)'%(0.55*st_f)
    wcol='rgba(255,255,255,%.2f)'%(0.12*st_f)
    cx=_lerp(0,1.6,st_f); cy=_lerp(0,1.6,st_f); cz=_lerp(2.5,1.2,st_f)
    svo=0.95*st_f; co=0.85*st_f
    R=60; xs=np.linspace(-7,7,R); ys=np.linspace(-7,7,R); Xg,Yg=np.meshgrid(xs,ys)
    Fval=w3[0]*Xg+w3[1]*Yg+w3[2]*zs*np.exp(-(Xg**2+Yg**2))+b3; famx=np.abs(Fval).max()
    r2_a=np.sum(a**2,axis=1); r2_b=np.sum(b**2,axis=1)
    fa=w3[0]*a[:,0]+w3[1]*a[:,1]+w3[2]*zs*np.exp(-r2_a)+b3
    fb=w3[0]*b[:,0]+w3[1]*b[:,1]+w3[2]*zs*np.exp(-r2_b)+b3
    svs=set()
    for sv in sv3: svs.add((round(sv[0],3),round(sv[1],3),round(sv[2],2)))
    ma=np.ones(n,dtype=bool); mb=np.ones(n,dtype=bool)
    for i in range(n):
        if (round(a[i,0],3),round(a[i,1],3),round(znr[i],2)) in svs: ma[i]=False
        if (round(b[i,0],3),round(b[i,1],3),round(znb[i],2)) in svs: mb[i]=False
    fig=go.Figure()
    fig.add_trace(go.Surface(x=Xg,y=Yg,z=np.zeros_like(Xg),surfacecolor=Fval,
        colorscale=[[0,'#A855F7'],[0.5,'#1E293B'],[1,'#22D3EE']],cmin=-famx,cmax=famx,showscale=True,
        colorbar=dict(title=dict(text='f(x)',side='right',font=dict(color='#94A3B8')),
                       tickfont=dict(color='#94A3B8'),len=0.5,y=0.25),opacity=0.65*st_f,
        name='f(x)',hovertemplate='f=%{surfacecolor:.2f}<extra></extra>',
        contours=dict(z=dict(show=True,color='rgba(255,255,255,0.4)',width=1))))
    fig.add_trace(go.Scatter3d(x=a[ma,0],y=a[ma,1],z=za[ma],mode='markers',name='類別 A',
        marker=dict(size=4,color=C_TEAL,opacity=0.85),
        customdata=fa[ma],hovertemplate='x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>f=%{customdata:.3f}<extra>A</extra>'))
    fig.add_trace(go.Scatter3d(x=b[mb,0],y=b[mb,1],z=zb[mb],mode='markers',name='類別 B',
        marker=dict(size=4,color=C_PURPLE,opacity=0.85),
        customdata=fb[mb],hovertemplate='x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>f=%{customdata:.3f}<extra>B</extra>'))
    if len(sv3)>0:
        sv_fa=w3[0]*sv3[:,0]+w3[1]*sv3[:,1]+w3[2]*sv3[:,2]+b3
        fig.add_trace(go.Scatter3d(x=sv3[:,0],y=sv3[:,1],z=zsv,mode='markers',name='3D SV',
            marker=dict(size=8,color=C_GOLD,opacity=svo,line=dict(width=2,color='#FFF'),symbol='diamond'),
            customdata=sv_fa,hovertemplate='SV f=%{customdata:.3f}<extra></extra>'))
    fig.add_trace(go.Surface(x=Xp,y=Yp,z=pz,
        colorscale=[[0,sc0],[0.48,wcol],[0.52,wcol],[1,sc1]],showscale=False,opacity=po,
        contours=dict(x=dict(show=True,color=cc,width=1),y=dict(show=True,color=cc,width=1)),
        name='f=0',hovertemplate='z=%{z:.1f}<extra>f=0</extra>'))
    if len(cv)>1:
        fig.add_trace(go.Scatter3d(x=cv[:,0],y=cv[:,1],z=np.zeros(len(cv)),mode='lines',
            name='邊界',line=dict(color=C_CURVE,width=3),opacity=co,hovertemplate='f=0<extra></extra>'))
    fig.update_layout(template="plotly_dark",paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(xaxis=dict(title='x₁',gridcolor=C_GRID,backgroundcolor='rgba(0,0,0,0)',range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
                   yaxis=dict(title='x₂',gridcolor=C_GRID,backgroundcolor='rgba(0,0,0,0)',range=[-7.5,7.5],title_font=dict(color='#94A3B8')),
                   zaxis=dict(title='Φ(z)',gridcolor=C_GRID,backgroundcolor='rgba(0,0,0,0)',range=[-0.5,zs+3],title_font=dict(color='#94A3B8')),
                   camera=dict(eye=dict(x=cx,y=cy,z=cz))),
        margin=dict(l=0,r=0,t=20,b=0),hovermode='closest',height=620,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0,font=dict(color='#94A3B8'),bgcolor='rgba(0,0,0,0)'))
    return fig

# ============================================================
CSS = """
<style>
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang TC','Microsoft YaHei',sans-serif}
.stApp{background:linear-gradient(160deg,#0A0A1A 0%,#0F1729 40%,#0C1222 100%)}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.92)!important;backdrop-filter:blur(16px);border-right:1px solid rgba(100,116,139,0.15)}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] span{color:#E2E8F0!important}
[data-testid="stSidebar"] .stCaption{color:#94A3B8!important}
[data-testid="stSidebar"] [data-testid="stButton"] button{padding:6px 2px!important;font-size:0.82rem!important;white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;width:100%!important;display:inline-flex!important;justify-content:center!important;align-items:center!important}
.card{background:rgba(15,23,42,0.75);border:1px solid rgba(100,116,139,0.18);border-radius:14px;padding:22px 26px;margin:14px 0;backdrop-filter:blur(12px);box-shadow:0 4px 24px rgba(0,0,0,0.3)}
.card h3{margin-top:0;font-weight:600;color:#E2E8F0}
.card p{color:#E2E8F0;line-height:1.7;font-size:0.95rem}
.main-title{font-size:2rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.02em;margin-bottom:4px}
.subtitle{font-size:0.95rem;color:#94A3B8;font-weight:400}
.legend-dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:middle}
.modebar{opacity:0.3}.modebar:hover{opacity:1}
</style>
"""

def main():
    try: _main()
    except Exception as e:
        st.set_page_config(page_title="SVM",page_icon="🔮",layout="wide")
        st.error(f"Error: {e}"); st.code(str(e))

def _main():
    st.set_page_config(page_title="SVM Kernel — 3D",page_icon="🔮",layout="wide")
    st.markdown(CSS,unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div style="padding:8px 0 16px"><div style="font-size:1.3rem;font-weight:700;color:#E2E8F0">🔮 SVM Kernel</div><div style="font-size:0.8rem;color:#64748B">3D Visualization</div></div>',unsafe_allow_html=True)
        st.markdown("### Params")
        np_val=st.slider("Particles/class",30,120,DEFAULT_N,10)
        zs_val=st.slider("Z-axis scale",2.0,15.0,DEFAULT_Z_SCALE,0.5)
        st.markdown("---"); st.markdown("### State")
        ci=st.session_state.get("_si",0); STATES=["Linear SVM","Nonlinear","Kernel 3D"]
        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("Linear",use_container_width=True,type="primary" if ci==0 else "secondary"): st.session_state.update(_si=0,_playing=True,_frame=0); st.rerun()
        with c2:
            if st.button("Nonlinear",use_container_width=True,type="primary" if ci==1 else "secondary"): st.session_state.update(_si=1,_playing=True,_frame=0); st.rerun()
        with c3:
            if st.button("3D",use_container_width=True,type="primary" if ci==2 else "secondary"): st.session_state.update(_si=2,_playing=True,_frame=0); st.rerun()
        st.markdown("---"); st.markdown("### Animation")
        manual_t = st.slider("Progress",0,100,0,1,key="_anim_slider",
                             help="Drag to scrub. Auto-plays on state switch.")
        playing = st.session_state.get("_playing",False)
        st.markdown("---")
        for c,l in [(C_TEAL,"Class A"),(C_PURPLE,"Class B"),(C_GOLD,"Support Vectors"),("#FFF","Decision Boundary"),(C_MARGIN,"Margins"),(C_CURVE,"Projection")]:
            st.markdown(f'<div style="display:flex;align-items:center;margin:5px 0;font-size:0.85rem;color:#94A3B8"><span class="legend-dot" style="background:{c};box-shadow:0 0 6px {c}44"></span>{l}</div>',unsafe_allow_html=True)

    st.session_state["_np"]=np_val; st.session_state["_zs"]=zs_val
    d=get_data(np_val,zs_val); w3,b3=d["w3"],d["b3"]; zs=d["z_scale"]

    # ---- auto-play logic ----
    # 首次載入：觸發自動播放
    if "_first_load" not in st.session_state:
        st.session_state["_first_load"] = True
        st.session_state["_playing"] = True
        st.session_state["_frame"] = 0
        st.rerun()

    playing = st.session_state.get("_playing", False)
    if playing:
        frame = st.session_state.get("_frame", 0)
        if frame < 100:
            frame = min(frame + 4, 100)
            st.session_state["_frame"] = frame
            t_val = frame / 100.0
            import time; time.sleep(0.15)
            st.rerun()
        else:
            st.session_state["_playing"] = False
            t_val = 1.0
    else:
        t_val = manual_t / 100.0

    st.markdown('<div class="main-title">SVM Kernel Trick: 3D Visualization</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">How SVMs use the kernel trick to map nonlinear data into higher dimensions</div>',unsafe_allow_html=True)

    sl=STATES[ci]
    info={
        "Linear SVM":("f(x) = wᵀx + b","Data is <b style='color:#22D3EE'>perfectly linearly separable</b>. SVM finds the <b style='color:#E2E8F0'>optimal hyperplane</b> maximizing margin.<br>White line = decision boundary, gold dashes = margins ±1, gold circles = support vectors.<br>💡 Drag the Animation slider to see elements appear."),
        "Nonlinear":("No linear separator in R²","Data <b style='color:#FBBF24'>cannot</b> be separated by a straight line. Teal cluster at center, purple ring around it.<br><b style='color:#A855F7'>Kernel trick</b> maps to higher-dimensional feature space.<br>💡 Drag slider to morph particles from linear to nonlinear positions."),
        "Kernel 3D":(f"Φ(x₁,x₂) = (x₁,x₂,{zs:.0f}·e<sup>−(x₁²+x₂²)</sup>)",f"Data <b style='color:#22D3EE'>lifted to 3D</b>!<br>Decision function f(x)=w·Φ(x)+b. Plane = f=0.<br><b style='color:#A855F7'>Floor heatmap</b>: purple=f&lt;0, teal=f&gt;0.<br>Gold diamonds = 3D support vectors. Green curve = 2D projection boundary.<br>💡 Drag slider to see z-lift + camera rotate. Hover for f(x) values."),
    }
    fi,fb=info[sl]
    st.markdown(f'<div class="card"><h3>{sl}</h3><p style="color:#FBBF24;font-family:monospace;font-size:1rem;">{fi}</p><p>{fb}</p></div>',unsafe_allow_html=True)

    try:
        if ci==0: fig=fig_linear(d,t_val)
        elif ci==1: fig=fig_nonlinear(d,t_val)
        else: fig=fig_kernel3d(d,t_val)
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':True,'displaylogo':False,'modeBarButtonsToRemove':['lasso2d','select2d','sendDataToCloud'],'scrollZoom':True})
    except Exception as e:
        st.error(f"Chart error: {e}"); st.code(str(e))

    if ci==2: st.caption(f"Note: intuitive visualization (z={zs:.0f}·exp(-(x²+y²))), not the true infinite-dimensional RBF feature space.")

if __name__=="__main__": main()
