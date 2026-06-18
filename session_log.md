# Session Log — VSM 3D Math Animation (SVM Kernel Visualization)

**Date:** 2026-06-18  
**Repo:** [miccowang66-max/VSM-3D-Math-animation](https://github.com/miccowang66-max/VSM-3D-Math-animation)  
**Deployed App:** https://vsm-3d-math-animation.streamlit.app/

---

## Issue 1 — Deployed App Kept Failing

### Problem
The Streamlit Community Cloud deployment at `https://vsm-3d-math-animation.streamlit.app/` was failing on every load with a rendering error.

### Root Cause
In `streamlit_app.py`, the 3D decision surface (`go.Surface`) in `build_kernel_3d_figure()` had invalid contour line widths:

```python
# BEFORE (invalid — Plotly requires width ∈ [1, 16])
contours=dict(
    x=dict(show=True, color='rgba(160,200,255,0.2)', width=0.5),
    y=dict(show=True, color='rgba(160,200,255,0.2)', width=0.5),
),
```

Plotly's `go.Surface` contour `width` parameter must be an integer between **1 and 16**. Passing `0.5` caused a validation exception that crashed the app on startup.

### Fix Applied
```python
# AFTER (valid)
contours=dict(
    x=dict(show=True, color='rgba(160,200,255,0.2)', width=1),
    y=dict(show=True, color='rgba(160,200,255,0.2)', width=1),
),
```

### Commit
- **Message:** `Fix Plotly surface contours width range error`
- **Hash:** `1f69b0f`
- **File:** `streamlit_app.py` (lines 464–467)

---

## Issue 2 — "3D" Sidebar Button Smaller Than Others / Poor Text Readability

### Problem
Two separate visual issues were reported after inspecting the deployed app:

1. **Button size mismatch** — The "🧊 3D" button in the sidebar was visually smaller than "📐 線性" and "🔄 非線性" because the emoji + short label caused inconsistent sizing in Streamlit's column layout.

2. **Low text contrast / poor readability** — Several text elements used colors that were too dark against the dark theme background:
   - Card body paragraph text: `#94A3B8` (too dim)
   - Page subtitle text: `#64748B` (very low contrast)
   - Sidebar legend labels: `#94A3B8` (hard to read)
   - Sidebar wildcard `* { color: #E2E8F0 }` was overriding slider/button colors incorrectly

### Fixes Applied

#### CSS — Sidebar Button Layout
Replaced the broad wildcard selector with targeted selectors and added button normalization to prevent text wrapping and enforce uniform size:

```css
/* Targeted sidebar text selectors instead of wildcard */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}

/* Button normalization — prevent wrapping, enforce uniform size */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    padding: 6px 2px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    width: 100% !important;
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
}
```

#### CSS — Text Contrast Improvements

| Element | Before | After |
|---------|--------|-------|
| `.card p` (body text) | `#94A3B8` | `#E2E8F0` |
| `.subtitle` | `#64748B` | `#94A3B8` |
| Sidebar legend labels | `#94A3B8` | `#E2E8F0` |
| Sidebar caption | wildcard override | `#94A3B8` (correct) |

### Commit
- **Message:** `Improve sidebar button layout and text contrast/readability`
- **Hash:** `3b82a2a`
- **File:** `streamlit_app.py` (CSS block + legend loop)

---

## Push to GitHub

All changes were pushed to the `master` branch of the GitHub repository. `.env` was confirmed to be excluded (already listed in `.gitignore`).

```
git push origin master
→ 1f69b0f..3b82a2a  master -> master
```

Streamlit Community Cloud automatically redeploys on every push to `master`.

---

## Files Modified

| File | Change |
|------|--------|
| `streamlit_app.py` | Fixed Plotly contour width; improved CSS selectors and button layout; boosted text contrast |

## Files Left Unchanged / Protected

| File | Reason |
|------|--------|
| `.env` | In `.gitignore` — never tracked or pushed |
| `requirements.txt` | No dependency changes needed |
| `CLAUDE.md` | Documentation only — not modified |

---

*Log generated: 2026-06-18 11:36 (CST)*
