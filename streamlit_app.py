"""
streamlit_app.py — SVM 核方法 3D 互動視覺化（動畫強化版）
========================================================
Plotly 原生動畫幀．粒子變形．Z軸提升．核映射過渡

執行: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ============================================================
# 常數 & 配色
# ============================================================
DEFAULT_Z_SCALE = 8.0
DEFAULT_N = 80
N_ANIM = 40

C_TEAL   = '#22D3EE'
C_PURPLE = '#A855F7'
C_GOLD   = '#FBBF24'
C_CURVE  = '#4ADE80'
C_MARGIN = '#FDE68A'
C_GRID   = 'rgba(100,116,139,0.12)'

# ============================================================
# 資料生成
# ============================================================

def _generate_datasets(n_per_class=DEFAULT_N, z_scale=DEFAULT_Z_SCALE):
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

    w_norm = w / np.linalg.norm(w)
    perp = np.array([-w_norm[1], w_norm[0]], dtype=np.float64)
    ext = 7.0; pc = -b_2d*w_norm
    p1 = pc + perp*ext; p2 = pc - perp*ext

    def _kz(pts): return z_scale * np.exp(-np.sum(pts**2, axis=1))
    def _phi(pts):
        r_sq = np.sum(pts**2, axis=1)
        return np.column_stack([pts, z_scale * np.exp(-r_sq)])

    phi_a = _phi(class_a_nonlin); phi_b = _phi(class_b_nonlin)
    znr = _kz(class_a_nonlin); znb = _kz(class_b_nonlin)

    red_c = np.mean(phi_a, axis=0); blue_c = np.mean(phi_b, axis=0)
    w3 = red_c - blue_c; w3 = w3 / np.linalg.norm(w3)
    mid3 = (red_c + blue_c) / 2.0; b3 = float(-np.dot(w3, mid3))

    da3 = np.abs(np.dot(phi_a, w3)+b3); db3 = np.abs(np.dot(phi_b, w3)+b3)
    sv_a = np.where(da3 <= np.percentile(da3, 10))[0]
    sv_b = np.where(db3 <= np.percentile(db3, 10))[0]
    sv_3d_phi = np.vstack([phi_a[sv_a], phi_b[sv_b]])

    curve_2d = _compute_boundary_curve(w3[0], w3[1], w3[2], b3, z_scale)
    plane_pts = _make_plane_surface(w3[0], w3[1], w3[2], b3, z_scale)

    return {
        "n": n_per_class, "z_scale": z_scale,
        "a_lin": class_a_lin, "b_lin": class_b_lin,
        "a_nl": class_a_nonlin, "b_nl": class_b_nonlin,
        "znr": znr, "znb": znb,
        "w": w, "b_2d": b_2d, "sv_2d": sv_2d, "p1": p1, "p2": p2,
        "w3": w3, "b3": b3, "sv_3d_phi": sv_3d_phi,
        "curve_2d": curve_2d, "plane_pts": plane_pts,
        "red_centroid_3d": red_c, "blue_centroid_3d": blue_c,
    }


def _compute_boundary_curve(a, b, c, d, z_scale, n_angles=150):
    pts = []
    for i in range(n_angles):
        theta = 2*np.pi*i/n_angles; ct, st = np.cos(theta), np.sin(theta)
        lo, hi = 0.0, 7.0
        flo = a*lo*ct + b*lo*st + c*z_scale*np.exp(-lo*lo) + d
        fhi = a*hi*ct + b*hi*st + c*z_scale*np.exp(-hi*hi) + d
        if flo*fhi > 0: continue
        for _ in range(30):
            mid = (lo+hi)/2
            fm = a*mid*ct + b*mid*st + c*z_scale*np.exp(-mid*mid) + d
            if abs(fm) < 0.005: break
            if flo*fm < 0: hi = mid; fhi = fm
            else: lo = mid; flo = fm
        r = (lo+hi)/2
        pts.append([float(r*ct), float(r*st)])
    return np.array(pts)


def _make_plane_surface(a, b, c, d, z_scale, extent=7.0, res=40):
    xs = np.linspace(-extent, extent, res)
    ys = np.linspace(-extent, extent, res)
    X, Y = np.meshgrid(xs, ys)
    Z = -(a*X + b*Y + d) / c if abs(c) > 1e-8 else np.zeros_like(X)
    Z = np.clip(Z, -2, z_scale + 4)
    return X, Y, Z


def get_data(n_particles=None, z_scale=None):
    key_n = st.session_state.get("_n_particles", DEFAULT_N)
    key_z = st.session_state.get("_z_scale", DEFAULT_Z_SCALE)
    n = n_particles if n_particles is not None else key_n
    z = z_scale if z_scale is not None else key_z
    ck = f"data_n{n}_z{z}"
    if ck not in st.session_state:
        st.session_state[ck] = _generate_datasets(n_per_class=n, z_scale=z)
    return st.session_state[ck]


def _lerp(a, b, t): return a + (b - a) * t


# ============================================================
# 動畫幀輔助
# ============================================================

def _anim_controls(label="▶ 播放"):
    return dict(
        updatemenus=[dict(
            type='buttons', showactive=False,
            x=0.05, y=-0.08, xanchor='left', yanchor='top',
            buttons=[
                dict(label=label, method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸ 暫停', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0, yanchor='top', xanchor='left',
            currentvalue=dict(prefix='幀: ', font=dict(color='#94A3B8')),
            transition=dict(duration=0), pad=dict(b=10, t=50),
            len=0.9, x=0.05, y=-0.02,
            steps=[dict(args=[[f'f{k}'], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                         label=str(k+1), method='animate') for k in range(N_ANIM)],
        )],
    )


# ============================================================
# 圖表建構（含動畫幀）
# ============================================================

def build_linear_figure(data):
    """線性 SVM — 三階段漸進動畫：粒子 → 決策邊界 → 邊界線 → SV"""
    a_lin, b_lin = data["a_lin"], data["b_lin"]
    sv_idx = set(data["sv_2d"]); n = data["n"]
    w, b_2d = data["w"], data["b_2d"]
    w_n = w / np.linalg.norm(w)

    ma = [i for i in range(n) if i not in sv_idx]
    mb = [i for i in range(n) if (i+n) not in sv_idx]
    sa = [i for i in range(n) if i in sv_idx]
    sb = [i-n for i in sv_idx if i >= n]

    perp = np.array([-w_n[1], w_n[0]]); pc = -b_2d*w_n; ext = 7.5
    xb = np.array([pc[0]-perp[0]*ext, pc[0]+perp[0]*ext])
    yb = np.array([pc[1]-perp[1]*ext, pc[1]+perp[1]*ext])

    fig = go.Figure()
    # trace 0-1: particles
    fig.add_trace(go.Scatter(x=a_lin[ma,0], y=a_lin[ma,1], mode='markers', name='類別 A',
        marker=dict(size=7, color=C_TEAL, opacity=0.75, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))))
    fig.add_trace(go.Scatter(x=b_lin[mb,0], y=b_lin[mb,1], mode='markers', name='類別 B',
        marker=dict(size=7, color=C_PURPLE, opacity=0.75, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))))
    # trace 2: boundary (opacity 0 → 1)
    fig.add_trace(go.Scatter(x=xb, y=yb, mode='lines', name='決策邊界',
        line=dict(color='#FFFFFF', width=2.2), opacity=0))
    # trace 3-4: margins
    for s in [1, -1]:
        o = s*w_n
        fig.add_trace(go.Scatter(x=xb+o[0], y=yb+o[1], mode='lines',
            name=f'{"+" if s>0 else "−"}1 Margin',
            line=dict(color=C_MARGIN, width=1.4, dash='dash'), opacity=0))
    # trace 5-6: SV
    fig.add_trace(go.Scatter(x=a_lin[sa,0], y=a_lin[sa,1], mode='markers', name='支持向量',
        marker=dict(size=1, color=C_GOLD, line=dict(width=1.5, color='rgba(255,255,255,0.3)'), symbol='circle-open'), opacity=0))
    fig.add_trace(go.Scatter(x=b_lin[sb,0], y=b_lin[sb,1], mode='markers', showlegend=False,
        marker=dict(size=1, color=C_GOLD, line=dict(width=1.5, color='rgba(255,255,255,0.3)'), symbol='circle-open'), opacity=0))

    frames = []
    for i in range(N_ANIM):
        t = (i+1)/N_ANIM
        bd = np.clip(t/0.3, 0, 1)
        mg = np.clip((t-0.2)/0.4, 0, 1)
        sv = np.clip((t-0.5)/0.5, 0, 1)
        frames.append(go.Frame(data=[
            None, None,
            dict(opacity=bd),
            dict(opacity=mg), dict(opacity=mg),
            dict(opacity=sv, marker=dict(size=1+sv*11)),
            dict(opacity=sv, marker=dict(size=1+sv*11)),
        ], name=f'f{i}'))
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID, zeroline=False, range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID, zeroline=False, range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest', height=620,
        **_anim_controls("▶ 播放動畫"),
    )
    return fig


def build_nonlinear_figure(data):
    """非線性資料 — 粒子從線性位置平滑變形到非線性位置"""
    a_lin, b_lin = data["a_lin"], data["b_lin"]
    a_nl, b_nl = data["a_nl"], data["b_nl"]
    n = data["n"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_lin[:,0], y=a_lin[:,1], mode='markers', name='類別 A',
        marker=dict(size=8, color=C_TEAL, opacity=0.8, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))))
    fig.add_trace(go.Scatter(x=b_lin[:,0], y=b_lin[:,1], mode='markers', name='類別 B',
        marker=dict(size=8, color=C_PURPLE, opacity=0.8, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))))
    # 示意環（後期出現）
    tc = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(x=4.25*np.cos(tc), y=4.25*np.sin(tc), mode='lines',
        showlegend=False, line=dict(color='rgba(168,85,247,0.25)', width=1.5, dash='dot'), opacity=0, hoverinfo='skip'))

    frames = []
    for i in range(N_ANIM):
        t = (i+1)/N_ANIM; et = 1 - (1-t)**3
        ax = _lerp(a_lin[:,0], a_nl[:,0], et)
        ay = _lerp(a_lin[:,1], a_nl[:,1], et)
        bx = _lerp(b_lin[:,0], b_nl[:,0], et)
        by = _lerp(b_lin[:,1], b_nl[:,1], et)
        ro = np.clip((t-0.7)/0.3, 0, 1)*0.7
        frames.append(go.Frame(data=[
            dict(x=ax, y=ay, marker=dict(size=8*(1-et)+5*et, color=C_TEAL, opacity=0.8, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))),
            dict(x=bx, y=by, marker=dict(size=8*(1-et)+5*et, color=C_PURPLE, opacity=0.8, line=dict(width=0.5, color='rgba(255,255,255,0.15)'))),
            dict(opacity=ro),
        ], name=f'f{i}'))
    fig.frames = frames

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID, zeroline=False, range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID, zeroline=False, range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest', height=620,
        **_anim_controls("▶ 播放變形"),
    )
    return fig


def build_kernel_3d_figure(data):
    """核方法 3D — Z軸提升 + 相機旋轉 + 決策平面出現"""
    a_nl, b_nl = data["a_nl"], data["b_nl"]
    znr, znb = data["znr"], data["znb"]
    sv3 = data["sv_3d_phi"]; curve = data["curve_2d"]
    Xp, Yp, Zp = data["plane_pts"]; n = data["n"]; zs = data["z_scale"]

    sv_set = set()
    for sv in sv3: sv_set.add((round(sv[0],3), round(sv[1],3), round(sv[2],2)))
    ma = np.ones(n, dtype=bool); mb = np.ones(n, dtype=bool)
    for i in range(n):
        if (round(a_nl[i,0],3), round(a_nl[i,1],3), round(znr[i],2)) in sv_set: ma[i] = False
        if (round(b_nl[i,0],3), round(b_nl[i,1],3), round(znb[i],2)) in sv_set: mb[i] = False

    fig = go.Figure()
    z0 = np.zeros(n)
    # trace 0-1: particles (z=0 initially)
    fig.add_trace(go.Scatter3d(x=a_nl[ma,0], y=a_nl[ma,1], z=z0[ma], mode='markers', name='類別 A',
        marker=dict(size=4, color=C_TEAL, opacity=0.8)))
    fig.add_trace(go.Scatter3d(x=b_nl[mb,0], y=b_nl[mb,1], z=z0[mb], mode='markers', name='類別 B',
        marker=dict(size=4, color=C_PURPLE, opacity=0.8)))
    # trace 2: SV
    sz0 = np.zeros(len(sv3)) if len(sv3) > 0 else []
    fig.add_trace(go.Scatter3d(
        x=sv3[:,0] if len(sv3)>0 else [], y=sv3[:,1] if len(sv3)>0 else [], z=sz0,
        mode='markers', name='3D SV',
        marker=dict(size=7, color=C_GOLD, opacity=0, line=dict(width=1.5,color='#FFFFFF'), symbol='diamond')))
    # trace 3: plane (invisible initially)
    fig.add_trace(go.Surface(x=Xp, y=Yp, z=np.zeros_like(Zp),
        colorscale=[[0,'rgba(180,210,255,0)'],[1,'rgba(180,210,255,0)']],
        showscale=False, opacity=0,
        contours=dict(x=dict(show=True,color='rgba(160,200,255,0)',width=0.5),
                      y=dict(show=True,color='rgba(160,200,255,0)',width=0.5)),
        name='SVM 決策平面'))
    # trace 4: projection curve
    if len(curve) > 1:
        fig.add_trace(go.Scatter3d(x=curve[:,0], y=curve[:,1], z=np.zeros(len(curve)),
            mode='lines', name='2D 邊界', line=dict(color=C_CURVE, width=2.5), opacity=0))

    frames = []
    for i in range(N_ANIM):
        t = (i+1)/N_ANIM; et = 1 - (1-t)**3
        zl = np.clip(et/0.5, 0, 1)
        za = znr*zl; zb = znb*zl; zsv = sv3[:,2]*zl if len(sv3)>0 else []
        st2 = np.clip((et-0.4)/0.6, 0, 1)
        st_f = float(st2)
        pz = Zp*st2; po = 0.45*st_f
        cc = 'rgba(160,200,255,{:.2f})'.format(0.25*st_f)
        sc0 = 'rgba(180,210,255,{:.2f})'.format(0.15*st_f)
        sc1 = 'rgba(180,210,255,{:.2f})'.format(0.38*st_f)
        co = 0.85*st_f; svo = 0.95*st_f
        cx = _lerp(0, 1.6, st_f); cy = _lerp(0, 1.6, st_f); cz = _lerp(2.5, 1.2, st_f)

        fd = [
            dict(z=za[ma]), dict(z=zb[mb]),
            dict(z=zsv, marker=dict(opacity=svo)) if len(sv3)>0 else {},
            dict(z=pz, opacity=po, colorscale=[[0,sc0],[1,sc1]],
                 contours=dict(x=dict(show=st_f>0.01,color=cc,width=0.5),
                              y=dict(show=st_f>0.01,color=cc,width=0.5))),
            dict(opacity=co) if len(curve)>1 else {},
        ]
        frames.append(go.Frame(data=fd, name=f'f{i}',
            layout=dict(scene=dict(camera=dict(eye=dict(x=cx,y=cy,z=cz))))))

    fig.frames = frames

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID, backgroundcolor='rgba(0,0,0,0)', range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
            yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID, backgroundcolor='rgba(0,0,0,0)', range=[-7.5,7.5], title_font=dict(color='#94A3B8')),
            zaxis=dict(title='Φ(z)', showgrid=True, gridcolor=C_GRID, backgroundcolor='rgba(0,0,0,0)', range=[-0.5,zs+3], title_font=dict(color='#94A3B8')),
            camera=dict(eye=dict(x=0, y=0, z=2.5)),
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest', height=650,
        **_anim_controls("▶ 播放核映射"),
    )
    return fig


# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang TC', 'Microsoft YaHei', sans-serif; }
.stApp { background: linear-gradient(160deg, #0A0A1A 0%, #0F1729 40%, #0C1222 100%); }
[data-testid="stSidebar"] { background: rgba(15,23,42,0.92) !important; backdrop-filter: blur(16px); border-right: 1px solid rgba(100,116,139,0.15); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] span { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stCaption { color: #94A3B8 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] button { padding: 6px 2px !important; font-size: 0.82rem !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; width: 100% !important; display: inline-flex !important; justify-content: center !important; align-items: center !important; }
.card { background: rgba(15,23,42,0.75); border: 1px solid rgba(100,116,139,0.18); border-radius: 14px; padding: 24px 28px; margin: 16px 0; backdrop-filter: blur(12px); box-shadow: 0 4px 24px rgba(0,0,0,0.3); }
.card h3 { margin-top: 0; font-weight: 600; color: #E2E8F0; }
.card p { color: #E2E8F0; line-height: 1.7; font-size: 0.95rem; }
.main-title { font-size: 2rem; font-weight: 700; color: #F1F5F9; letter-spacing: -0.02em; margin-bottom: 4px; }
.subtitle { font-size: 0.95rem; color: #94A3B8; font-weight: 400; }
.legend-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
.modebar { opacity: 0.3; } .modebar:hover { opacity: 1; }
</style>
"""

# ============================================================
# STREAMLIT 主頁面
# ============================================================

def main():
    try:
        _main_impl()
    except Exception as e:
        st.error(f"🚨 應用程式啟動失敗：{e}")
        st.code(str(e))
        import traceback
        st.code(traceback.format_exc())


def _main_impl():
    st.set_page_config(page_title="SVM 核方法 — 3D 視覺化", page_icon="🔮", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ---- 側邊欄 ----
    with st.sidebar:
        st.markdown("""<div style="padding:8px 0 16px 0;"><div style="font-size:1.3rem;font-weight:700;color:#E2E8F0;">🔮 SVM 核方法</div><div style="font-size:0.8rem;color:#64748B;">Kernel Trick · 3D 視覺化</div></div>""", unsafe_allow_html=True)

        st.markdown("### 🎛 參數控制")
        n_particles = st.slider("粒子數量（每類）", 30, 120, DEFAULT_N, 10, help="調整每類別生成的資料點數量")
        z_scale = st.slider("Z 軸放大倍率", 2.0, 15.0, DEFAULT_Z_SCALE, 0.5, help="核映射 Φ(z) 的視覺放大倍數")

        st.markdown("---"); st.markdown("### 📍 狀態選擇")
        STATES = ["線性 SVM", "非線性資料", "核方法 3D"]
        ci = st.session_state.get("_state_idx", 0)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📐 線性", use_container_width=True, type="primary" if ci==0 else "secondary", key="bl"):
                st.session_state["_state_idx"] = 0; st.rerun()
        with c2:
            if st.button("🔄 非線性", use_container_width=True, type="primary" if ci==1 else "secondary", key="bn"):
                st.session_state["_state_idx"] = 1; st.rerun()
        with c3:
            if st.button("🧊 3D", use_container_width=True, type="primary" if ci==2 else "secondary", key="bk"):
                st.session_state["_state_idx"] = 2; st.rerun()

        st.markdown("---"); st.markdown("### 🎨 圖例")
        for c, l in [(C_TEAL,"類別 A"),(C_PURPLE,"類別 B"),(C_GOLD,"支持向量"),("#FFFFFF","決策邊界"),(C_MARGIN,"邊界線 ±1"),(C_CURVE,"投影曲線")]:
            st.markdown(f'<div style="display:flex;align-items:center;margin:6px 0;font-size:0.85rem;color:#94A3B8;"><span class="legend-dot" style="background:{c};box-shadow:0 0 6px {c}44;"></span>{l}</div>', unsafe_allow_html=True)
        st.markdown("---"); st.caption(f"n={n_particles}, z={z_scale:.1f}")
        st.caption("💡 點擊圖表下方 ▶ 按鈕播放動畫")

    # ---- 資料 ----
    st.session_state["_n_particles"] = n_particles
    st.session_state["_z_scale"] = z_scale
    data = get_data(n_particles, z_scale)
    w3, b3 = data["w3"], data["b3"]; zs = data["z_scale"]

    # ---- 主內容 ----
    st.markdown('<div class="main-title">SVM 核方法：3D 互動視覺化</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">展示支持向量機如何透過核方法將非線性資料映射到高維空間，使其線性可分</div>', unsafe_allow_html=True)

    info = {
        "線性 SVM": ("f(x) = wᵀx + b",
            "資料 <b style='color:#22D3EE'>完美線性可分</b>。SVM 找到最大化邊界的 <b style='color:#E2E8F0'>最佳超平面</b>。<br>白色實線為決策邊界 wᵀx+b=0，金黃虛線為邊界 wᵀx+b=±1，金色空心圓為支持向量。<br><i style='color:#64748B;font-size:0.85rem;'>💡 點擊 ▶ 按鈕觀看動畫</i>"),
        "非線性資料": ("不存在線性分隔器 (ℝ²)",
            "資料 <b style='color:#FBBF24'>無法</b> 用一條直線分開。碧藍群聚於中心，霓虹紫分布於外環。<br><b style='color:#A855F7'>核方法</b> 將其映射到更高維度的特徵空間。<br><i style='color:#64748B;font-size:0.85rem;'>💡 點擊 ▶ 觀看粒子變形動畫，再切換到 3D</i>"),
        "核方法 3D": (f"Φ(x₁,x₂) = (x₁, x₂, {zs:.0f}·e<sup>−(x₁²+x₂²)</sup>)",
            f"資料經核函數 <b style='color:#22D3EE'>提升到 3D 特徵空間</b>！SVM 決策平面：{w3[0]:.2f}x₁ + {w3[1]:.2f}x₂ + {w3[2]:.2f}z + {b3:.2f} = 0<br>半透明面為 <b style='color:#E2E8F0'>3D 決策平面</b>，綠色曲線為 <b style='color:#4ADE80'>2D 投影邊界</b>。<br><i style='color:#64748B;font-size:0.85rem;'>💡 點擊 ▶ 觀看核映射動畫，拖曳旋轉/縮放圖表</i>"),
    }
    sl = STATES[ci]
    fi, fb = info[sl]
    st.markdown(f"""<div class="card"><h3>{sl}</h3><p style="color:#FBBF24;font-family:monospace;font-size:1rem;">{fi}</p><p>{fb}</p></div>""", unsafe_allow_html=True)

    # ---- 圖表 ----
    try:
        if ci == 0: fig = build_linear_figure(data)
        elif ci == 1: fig = build_nonlinear_figure(data)
        else: fig = build_kernel_3d_figure(data)
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True, 'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d','select2d','sendDataToCloud'],
            'scrollZoom': True,
        })
    except Exception as e:
        st.error(f"圖表渲染失敗：{e}"); st.code(str(e))

    if ci == 2:
        st.caption(f"⚠ 此為特徵空間提升的直觀示意（z = {zs:.0f}·e<sup>−(x²+y²)</sup>），並非真實 RBF 核的無限維映射。")


if __name__ == "__main__":
    main()
