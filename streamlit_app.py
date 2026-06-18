"""
streamlit_app.py — SVM 核方法 3D 互動視覺化（精緻設計版）
========================================================
現代化深色主題．莫蘭迪配色．Plotly 原生 3D 渲染．無 CDN 依賴

執行: streamlit run streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# ============================================================
# 常數 & 配色
# ============================================================
DEFAULT_Z_SCALE = 8.0
DEFAULT_N = 80
C_TEAL   = '#22D3EE'
C_TEAL_BRIGHT = '#67E8F9'
C_PURPLE = '#A855F7'
C_PURPLE_BRIGHT = '#C084FC'
C_GOLD   = '#FBBF24'
C_SURFACE = 'rgba(200,220,255,0.32)'
C_SURFACE_LINE = 'rgba(160,200,255,0.55)'
C_CURVE  = '#4ADE80'
C_MARGIN = '#FDE68A'
C_AXIS   = '#475569'
C_GRID   = 'rgba(100,116,139,0.12)'

# ============================================================
# 資料生成（數學邏輯不變，接受參數以支援互動）
# ============================================================

def _generate_datasets(n_per_class=DEFAULT_N, z_scale=DEFAULT_Z_SCALE):
    """生成線性 + 非線性資料集，計算 SVM 決策面參數。"""
    rng = np.random.RandomState(42)

    # 線性資料
    class_a_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([-3, -3])
    class_b_lin = rng.randn(n_per_class, 2) * 1.2 + np.array([3, 3])
    X_linear = np.vstack([class_a_lin, class_b_lin])
    y_linear = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # 非線性資料
    class_a_nonlin = rng.randn(n_per_class, 2) * 1.0
    angles = rng.uniform(0, 2*np.pi, n_per_class)
    radii = rng.uniform(3.5, 5.0, n_per_class)
    class_b_nonlin = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

    # 線性 SVM
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

    # 邊界線幾何
    w_norm = w / np.linalg.norm(w)
    perp = np.array([-w_norm[1], w_norm[0]], dtype=np.float64)
    ext = 7.0; pc = -b_2d*w_norm
    p1 = pc + perp*ext; p2 = pc - perp*ext

    # 核函數
    def _phi(pts):
        r_sq = np.sum(pts**2, axis=1)
        return np.column_stack([pts, z_scale * np.exp(-r_sq)])

    def _kz(pts):
        return z_scale * np.exp(-np.sum(pts**2, axis=1))

    phi_a = _phi(class_a_nonlin)
    phi_b = _phi(class_b_nonlin)
    znr = _kz(class_a_nonlin)
    znb = _kz(class_b_nonlin)

    # 3D 最大邊界決策平面
    red_c = np.mean(phi_a, axis=0)
    blue_c = np.mean(phi_b, axis=0)
    w3 = red_c - blue_c
    w3 = w3 / np.linalg.norm(w3)
    mid3 = (red_c + blue_c) / 2.0
    b3 = float(-np.dot(w3, mid3))

    # 3D SV
    da3 = np.abs(np.dot(phi_a, w3)+b3); db3 = np.abs(np.dot(phi_b, w3)+b3)
    sv_a = np.where(da3 <= np.percentile(da3, 10))[0]
    sv_b = np.where(db3 <= np.percentile(db3, 10))[0]
    sv_3d_phi = np.vstack([phi_a[sv_a], phi_b[sv_b]])

    # 2D 投影邊界曲線
    curve_2d = _compute_boundary_curve(w3[0], w3[1], w3[2], b3, z_scale)

    # 3D 平面網格
    plane_pts = _make_plane_surface(w3[0], w3[1], w3[2], b3, z_scale)

    # 線性 SV 座標
    sv_linear_pts = np.vstack([X_linear[sv_2d]])

    return {
        "n": n_per_class,
        "z_scale": z_scale,
        "a_lin": class_a_lin, "b_lin": class_b_lin,
        "a_nl": class_a_nonlin, "b_nl": class_b_nonlin,
        "znr": znr, "znb": znb,
        "w": w, "b_2d": b_2d,
        "sv_2d": sv_2d, "sv_linear_pts": sv_linear_pts,
        "p1": p1, "p2": p2,
        "w3": w3, "b3": b3,
        "sv_3d_phi": sv_3d_phi,
        "curve_2d": curve_2d,
        "plane_pts": plane_pts,
        "red_centroid_3d": red_c,
        "blue_centroid_3d": blue_c,
    }


def _compute_boundary_curve(a, b, c, d, z_scale=DEFAULT_Z_SCALE, n_angles=150):
    """取樣 2D 決策邊界曲線（二分搜尋半徑）。"""
    pts = []
    for i in range(n_angles):
        theta = 2*np.pi*i/n_angles
        ct, st = np.cos(theta), np.sin(theta)
        lo, hi = 0.0, 7.0
        flo = a*lo*ct + b*lo*st + c*z_scale*np.exp(-lo*lo) + d
        fhi = a*hi*ct + b*hi*st + c*z_scale*np.exp(-hi*hi) + d
        if flo*fhi > 0:
            continue
        for _ in range(30):
            mid = (lo+hi)/2
            fm = a*mid*ct + b*mid*st + c*z_scale*np.exp(-mid*mid) + d
            if abs(fm) < 0.005:
                break
            if flo*fm < 0:
                hi = mid; fhi = fm
            else:
                lo = mid; flo = fm
        r = (lo+hi)/2
        pts.append([float(r*ct), float(r*st)])
    return np.array(pts)


def _make_plane_surface(a, b, c, d, z_scale=DEFAULT_Z_SCALE, extent=7.0, res=40):
    """在 XY 網格上計算決策平面的 Z 值。"""
    xs = np.linspace(-extent, extent, res)
    ys = np.linspace(-extent, extent, res)
    X, Y = np.meshgrid(xs, ys)
    if abs(c) > 1e-8:
        Z = -(a*X + b*Y + d) / c
    else:
        Z = np.zeros_like(X)
    Z = np.clip(Z, -2, z_scale + 4)
    return X, Y, Z


# ============================================================
# 一次性生成資料（session_state 快取）
# ============================================================

def get_data(n_particles=None, z_scale=None):
    """Get or regenerate data. Regenerates if parameters changed."""
    key_n = st.session_state.get("_n_particles", DEFAULT_N)
    key_z = st.session_state.get("_z_scale", DEFAULT_Z_SCALE)
    n = n_particles if n_particles is not None else key_n
    z = z_scale if z_scale is not None else key_z
    cache_key = f"data_n{n}_z{z}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _generate_datasets(n_per_class=n, z_scale=z)
    return st.session_state[cache_key]


# ============================================================
# STREAMLIT 自訂 CSS
# ============================================================

CUSTOM_CSS = """
<style>
/* 全域字體與背景 */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: linear-gradient(160deg, #0A0A1A 0%, #0F1729 40%, #0C1222 100%);
}

/* 側邊欄 */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.92) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(100,116,139,0.15);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* 卡片容器 */
.card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(100,116,139,0.18);
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.card h3 { margin-top: 0; font-weight: 600; color: #E2E8F0; }
.card p { color: #94A3B8; line-height: 1.7; font-size: 0.95rem; }

/* 標題 */
.main-title {
    font-size: 2rem; font-weight: 700; color: #F1F5F9;
    letter-spacing: -0.02em; margin-bottom: 4px;
}
.subtitle {
    font-size: 0.95rem; color: #64748B; font-weight: 400;
}

/* Legend */
.legend-dot {
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; margin-right: 8px; vertical-align: middle;
}
.legend-line {
    display: inline-block; width: 20px; height: 3px;
    border-radius: 2px; margin-right: 8px; vertical-align: middle;
}

/* Segmented control styling */
div[data-testid="stRadio"] > div {
    gap: 4px;
}
div[data-testid="stRadio"] label {
    padding: 10px 18px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(100,116,139,0.2) !important;
    background: rgba(30,41,59,0.6) !important;
    transition: all 0.2s;
    margin-bottom: 6px !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: rgba(34,211,238,0.4) !important;
    background: rgba(34,211,238,0.08) !important;
}
div[data-testid="stRadio"] label[data-selected="true"] {
    border-color: #22D3EE !important;
    background: rgba(34,211,238,0.15) !important;
    color: #22D3EE !important;
    font-weight: 600;
}

/* 隱藏 Plotly modebar 中不需要的按鈕 */
.modebar { opacity: 0.3; }
.modebar:hover { opacity: 1; }
</style>
"""

# ============================================================
# 圖表建構函式
# ============================================================

def build_linear_figure(data):
    """線性 SVM — 2D 散點圖 + 決策邊界線 + 邊界線 + SV"""
    fig = go.Figure()

    a_lin, b_lin = data["a_lin"], data["b_lin"]
    sv_idx = set(data["sv_2d"])
    n = data["n"]
    w, b_2d = data["w"], data["b_2d"]
    w_n = w / np.linalg.norm(w)

    # 非 SV 的紅點
    mask_a_nosv = [i for i in range(n) if i not in sv_idx]
    mask_b_nosv = [i for i in range(n) if (i+n) not in sv_idx]

    fig.add_trace(go.Scatter(
        x=a_lin[mask_a_nosv, 0], y=a_lin[mask_a_nosv, 1],
        mode='markers', name='類別 A',
        marker=dict(size=7, color=C_TEAL, opacity=0.75,
                     line=dict(width=0.5, color='rgba(255,255,255,0.15)')),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=b_lin[mask_b_nosv, 0], y=b_lin[mask_b_nosv, 1],
        mode='markers', name='類別 B',
        marker=dict(size=7, color=C_PURPLE, opacity=0.75,
                     line=dict(width=0.5, color='rgba(255,255,255,0.15)')),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    ))

    # SV 標示
    sv_a = [i for i in range(n) if i in sv_idx]
    sv_b = [i-n for i in sv_idx if i >= n]
    fig.add_trace(go.Scatter(
        x=a_lin[sv_a, 0], y=a_lin[sv_a, 1],
        mode='markers', name='支持向量',
        marker=dict(size=12, color=C_GOLD, opacity=0.95,
                     line=dict(width=2, color='rgba(255,255,255,0.5)'),
                     symbol='circle-open'),
        hovertemplate='SV-A<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=b_lin[sv_b, 0], y=b_lin[sv_b, 1],
        mode='markers', showlegend=False,
        marker=dict(size=12, color=C_GOLD, opacity=0.95,
                     line=dict(width=2, color='rgba(255,255,255,0.5)'),
                     symbol='circle-open'),
        hovertemplate='SV-B<extra></extra>',
    ))

    # 決策邊界線: w·x + b = 0，取兩端點
    perp = np.array([-w_n[1], w_n[0]])
    pc = -b_2d * w_n
    ext = 7.5
    xb = np.array([pc[0] - perp[0]*ext, pc[0] + perp[0]*ext])
    yb = np.array([pc[1] - perp[1]*ext, pc[1] + perp[1]*ext])
    fig.add_trace(go.Scatter(
        x=xb, y=yb, mode='lines', name='決策邊界',
        line=dict(color='#FFFFFF', width=2.2, dash='solid'),
        hovertemplate='wᵀx+b=0<extra></extra>',
    ))

    # 邊界線 ±1
    for sign, name, dash in [(1, '+1 Margin', 'dash'), (-1, '−1 Margin', 'dash')]:
        offset = sign * w_n
        xbm = xb + offset[0]
        ybm = yb + offset[1]
        fig.add_trace(go.Scatter(
            x=xbm, y=ybm, mode='lines', name=name,
            line=dict(color=C_MARGIN, width=1.4, dash=dash),
            opacity=0.7, hovertemplate='wᵀx+b=%{text}<extra></extra>',
            text=[f'{sign}' for _ in xbm],
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID, zeroline=False,
                    range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID, zeroline=False,
                    range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                     font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
        height=620,
    )
    return fig


def build_nonlinear_figure(data):
    """非線性資料 — 2D 散點，紅中心藍外環，無線性分隔器。"""
    fig = go.Figure()
    a_nl, b_nl = data["a_nl"], data["b_nl"]

    fig.add_trace(go.Scatter(
        x=a_nl[:, 0], y=a_nl[:, 1],
        mode='markers', name='類別 A',
        marker=dict(size=8, color=C_TEAL, opacity=0.8,
                     line=dict(width=0.5, color='rgba(255,255,255,0.15)')),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=b_nl[:, 0], y=b_nl[:, 1],
        mode='markers', name='類別 B',
        marker=dict(size=8, color=C_PURPLE, opacity=0.8,
                     line=dict(width=0.5, color='rgba(255,255,255,0.15)')),
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
    ))

    # 淡色虛線圓環（示意）
    theta_c = np.linspace(0, 2*np.pi, 200)
    r_mid = 4.25
    fig.add_trace(go.Scatter(
        x=r_mid*np.cos(theta_c), y=r_mid*np.sin(theta_c),
        mode='lines', name='環形結構', showlegend=False,
        line=dict(color='rgba(168,85,247,0.2)', width=1, dash='dot'),
        hoverinfo='skip',
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID, zeroline=False,
                    range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
        yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID, zeroline=False,
                    range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                     font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
        height=620,
    )
    return fig


def build_kernel_3d_figure(data):
    """核方法 3D — 3D 散點 + SVM 決策平面 + 投影邊界曲線。"""
    fig = go.Figure()
    a_nl, b_nl = data["a_nl"], data["b_nl"]
    znr, znb = data["znr"], data["znb"]
    sv3 = data["sv_3d_phi"]
    curve = data["curve_2d"]
    Xp, Yp, Zp = data["plane_pts"]
    n = data["n"]
    zs = data["z_scale"]

    # 建立 SV 集合
    sv_set = set()
    for sv in sv3:
        sv_set.add((round(sv[0], 3), round(sv[1], 3), round(sv[2], 2)))

    mask_a = np.ones(n, dtype=bool)
    mask_b = np.ones(n, dtype=bool)
    for i in range(n):
        if (round(a_nl[i,0],3), round(a_nl[i,1],3), round(znr[i],2)) in sv_set:
            mask_a[i] = False
        if (round(b_nl[i,0],3), round(b_nl[i,1],3), round(znb[i],2)) in sv_set:
            mask_b[i] = False

    # Class A
    fig.add_trace(go.Scatter3d(
        x=a_nl[mask_a, 0], y=a_nl[mask_a, 1], z=znr[mask_a],
        mode='markers', name='類別 A',
        marker=dict(size=4, color=C_TEAL, opacity=0.8),
        hovertemplate='(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>',
    ))
    # Class B
    fig.add_trace(go.Scatter3d(
        x=b_nl[mask_b, 0], y=b_nl[mask_b, 1], z=znb[mask_b],
        mode='markers', name='類別 B',
        marker=dict(size=4, color=C_PURPLE, opacity=0.8),
        hovertemplate='(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>',
    ))

    # SV 3D
    if len(sv3) > 0:
        fig.add_trace(go.Scatter3d(
            x=sv3[:, 0], y=sv3[:, 1], z=sv3[:, 2],
            mode='markers', name='3D SV',
            marker=dict(size=7, color=C_GOLD, opacity=0.95,
                         line=dict(width=1.5, color='#FFFFFF'),
                         symbol='diamond'),
            hovertemplate='SV<extra></extra>',
        ))

    # 決策平面 (Surface) — 最小化參數以確保相容性
    fig.add_trace(go.Surface(
        x=Xp, y=Yp, z=Zp,
        colorscale=[[0, 'rgba(180,210,255,0.18)'], [1, 'rgba(180,210,255,0.42)']],
        showscale=False,
        opacity=0.6,
        contours=dict(
            x=dict(show=True, color='rgba(160,200,255,0.2)', width=0.5),
            y=dict(show=True, color='rgba(160,200,255,0.2)', width=0.5),
        ),
        name='SVM 決策平面',
        hovertemplate='z=%{z:.1f}<extra>決策面</extra>',
    ))

    # 地板投影曲線
    if len(curve) > 1:
        fig.add_trace(go.Scatter3d(
            x=curve[:, 0], y=curve[:, 1], z=np.zeros(len(curve)),
            mode='lines', name='2D 邊界',
            line=dict(color=C_CURVE, width=2.5),
            hovertemplate='投影邊界<extra></extra>',
        ))

    # 場景設定
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='x₁', showgrid=True, gridcolor=C_GRID,
                        backgroundcolor='rgba(0,0,0,0)',
                        range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
            yaxis=dict(title='x₂', showgrid=True, gridcolor=C_GRID,
                        backgroundcolor='rgba(0,0,0,0)',
                        range=[-7.5, 7.5], title_font=dict(color='#94A3B8')),
            zaxis=dict(title='Φ(z)', showgrid=True, gridcolor=C_GRID,
                        backgroundcolor='rgba(0,0,0,0)',
                        range=[-0.5, zs+3], title_font=dict(color='#94A3B8')),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                     font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        hovermode='closest',
        height=650,
    )
    return fig


# ============================================================
# STREAMLIT 主頁面
# ============================================================

def main():
    st.set_page_config(
        page_title="SVM 核方法 — 3D 視覺化",
        page_icon="🔮",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ============================================================
    # 側邊欄 — 互動控制
    # ============================================================
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 16px 0;">
            <div style="font-size:1.3rem;font-weight:700;color:#E2E8F0;">🔮 SVM 核方法</div>
            <div style="font-size:0.8rem;color:#64748B;">Kernel Trick · 3D 視覺化</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎛 參數控制")
        n_particles = st.slider(
            "粒子數量（每類）", min_value=30, max_value=120,
            value=DEFAULT_N, step=10,
            help="調整每類別生成的資料點數量",
        )
        z_scale = st.slider(
            "Z 軸放大倍率", min_value=2.0, max_value=15.0,
            value=DEFAULT_Z_SCALE, step=0.5,
            help="核映射 Φ(z) 的視覺放大倍數，越大代表 3D 分離越明顯",
        )

        st.markdown("---")
        st.markdown("### 📍 狀態選擇")

        STATES = ["線性 SVM", "非線性資料", "核方法 3D"]
        STATE_KEYS = ["linear", "nonlinear", "kernel3d"]

        # 讀取目前狀態
        current_idx = st.session_state.get("_state_idx", 0)
        state_label = STATES[current_idx]

        # 三個按鈕橫排
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📐 線性", use_container_width=True,
                         type="primary" if current_idx == 0 else "secondary",
                         key="btn_linear"):
                st.session_state["_state_idx"] = 0
                st.rerun()
        with c2:
            if st.button("🔄 非線性", use_container_width=True,
                         type="primary" if current_idx == 1 else "secondary",
                         key="btn_nonlinear"):
                st.session_state["_state_idx"] = 1
                st.rerun()
        with c3:
            if st.button("🧊 3D", use_container_width=True,
                         type="primary" if current_idx == 2 else "secondary",
                         key="btn_kernel3d"):
                st.session_state["_state_idx"] = 2
                st.rerun()

        # 互動提示
        st.markdown("---")
        st.caption("💡 試試調整上方滑桿，改變粒子數量與 Z 軸倍率。")
        st.caption("🖱 在圖表上拖曳可旋轉/縮放/平移。")

        st.markdown("---")
        st.markdown("### 🎨 圖例")
        legend_items = [
            (C_TEAL, "類別 A"),
            (C_PURPLE, "類別 B"),
            (C_GOLD, "支持向量"),
            ("#FFFFFF", "決策邊界"),
            (C_MARGIN, "邊界線 ±1"),
            (C_CURVE, "投影曲線"),
        ]
        for color, label in legend_items:
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:6px 0;font-size:0.85rem;color:#94A3B8;">'
                f'<span class="legend-dot" style="background:{color};box-shadow:0 0 6px {color}44;"></span>{label}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(f"目前參數：n={n_particles}, z={z_scale:.1f}")

    # ============================================================
    # 資料載入（參數改變時自動重建）
    # ============================================================
    st.session_state["_n_particles"] = n_particles
    st.session_state["_z_scale"] = z_scale
    data = get_data(n_particles, z_scale)
    w3, b3 = data["w3"], data["b3"]
    zs = data["z_scale"]

    # ============================================================
    # 主內容區
    # ============================================================
    st.markdown('<div class="main-title">SVM 核方法：3D 互動視覺化</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">展示支持向量機如何透過核方法將非線性資料映射到高維空間，使其線性可分</div>',
        unsafe_allow_html=True,
    )

    # ---- 資訊卡片 ----
    info_map = {
        "線性 SVM": {
            "formula": "f(x) = wᵀx + b",
            "body": (
                "資料 <b style='color:#22D3EE'>完美線性可分</b>。"
                "SVM 找到最大化邊界的 <b style='color:#E2E8F0'>最佳超平面</b>。<br>"
                "白色實線為決策邊界 wᵀx+b=0，金黃虛線為邊界 wᵀx+b=±1，金色空心圓為支持向量。<br>"
                "<i style='color:#64748B;font-size:0.85rem;'>💡 拖曳滑鼠可縮放及平移圖表</i>"
            ),
        },
        "非線性資料": {
            "formula": "不存在線性分隔器 (ℝ²)",
            "body": (
                "資料 <b style='color:#FBBF24'>無法</b> 用一條直線分開。"
                "紅色（碧藍）群聚於中心，藍色（霓虹紫）分布於外環。<br>"
                "<b style='color:#A855F7'>核方法</b> 將其映射到更高維度的特徵空間，使其變得可線性分離。<br>"
                "<i style='color:#64748B;font-size:0.85rem;'>💡 點擊「3D」按鈕查看核方法效果</i>"
            ),
        },
        "核方法 3D": {
            "formula": f"Φ(x₁,x₂) = (x₁, x₂, {zs:.0f}·e<sup>−(x₁²+x₂²)</sup>)",
            "body": (
                f"資料經核函數 <b style='color:#22D3EE'>提升到 3D 特徵空間</b>！<br>"
                f"SVM 決策平面：{w3[0]:.2f}x₁ + {w3[1]:.2f}x₂ + {w3[2]:.2f}z + {b3:.2f} = 0<br><br>"
                f"半透明面為 <b style='color:#E2E8F0'>3D 決策平面</b>，"
                f"綠色曲線為其 <b style='color:#4ADE80'>2D 投影邊界</b>。<br>"
                f"金色菱形為 3D 支持向量。<br>"
                f"<i style='color:#64748B;font-size:0.85rem;'>💡 拖曳旋轉、滾輪縮放、右鍵平移。試試調整左側 Z 軸倍率滑桿！</i>"
            ),
        },
    }
    info = info_map[STATES[current_idx]]
    st.markdown(
        f"""
        <div class="card">
            <h3>{STATES[current_idx]}</h3>
            <p style="color:#FBBF24;font-family:monospace;font-size:1rem;">{info['formula']}</p>
            <p>{info['body']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 圖表 ----
    try:
        if current_idx == 0:
            fig = build_linear_figure(data)
        elif current_idx == 1:
            fig = build_nonlinear_figure(data)
        else:
            fig = build_kernel_3d_figure(data)

        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'sendDataToCloud'],
            'displaylogo': False,
            'scrollZoom': True,
        })
    except Exception as e:
        st.error(f"圖表渲染失敗：{e}")
        st.code(str(e))

    # 底部提示
    if current_idx == 2:
        st.caption(
            f"⚠ 此為特徵空間提升的直觀示意（z = {zs:.0f}·e<sup>−(x²+y²)</sup>），"
            "並非真實 RBF 核的無限維映射。拖曳圖表可旋轉視角、檢視三維分離結構。"
        )


if __name__ == "__main__":
    main()
