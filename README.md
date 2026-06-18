# VSM 3D Math Animation — SVM Kernel Trick Visualization

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Three.js](https://img.shields.io/badge/Three.js-r160-black?style=flat-square&logo=threedotjs)](https://threejs.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> An interactive 3D educational visualization demonstrating how **Support Vector Machines** use the **kernel trick** to handle nonlinearly separable data by lifting it into higher dimensions. Built with **Three.js · Streamlit · scikit-learn**.

---

## 🌐 Live Demo

| Platform | URL | Description |
|---|---|---|
| **Streamlit Cloud** | [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io) | One-click deploy to Streamlit Community Cloud |

> **Try it locally in 30 seconds:** `pip install -r requirements.txt && streamlit run streamlit_app.py`

---

## 🎯 What You'll See

### Three Interactive States

| State | Description | Visual |
|---|---|---|
| **LINEAR SVM** | Perfectly separable data with optimal hyperplane, margins, and support vectors | `LINEAR →` |
| **NONLINEAR DATA** | Same particles morphed — Red cluster at center, Blue ring around it. No straight line can separate them | `NONLINEAR →` |
| **KERNEL 3D** | Data lifted to 3D via Φ(x₁,x₂) = (x₁, x₂, e^−(x₁²+x₂²)). A hyperplane now separates them easily | `KERNEL_3D` |

**In the KERNEL_3D state, you can drag to rotate, scroll to zoom, and right-drag to pan.**

### Scene Architecture (6 Layers)

| Layer | Contents |
|---|---|
| **Background** | 1,800-star starfield + nebula particles — always visible |
| **Data** | Class A (Neon Red) + Class B (Neon Blue), BufferGeometry + PointsMaterial |
| **Decision Boundary** | Linear separator + margin lines (wᵀx+b=±1) + support vector glow rings |
| **Kernel Transformation** | 28×28 grid warping via z = exp(−(x²+y²)) |
| **Hyperplane** | Translucent separation plane at computed z-height + orbiting glow ring |
| **Projection** | Circular decision boundary projected onto the floor + vertical projection lines |

---

## ✨ Features

- 🔮 **Three.js 3D Engine** — WebGL rendering with EffectComposer + UnrealBloomPass post-processing
- 🎛 **Finite State Machine** — LINEAR → NONLINEAR → KERNEL_3D with 2.8s eased transitions
- 🧮 **Real SVM Computation** — Support vectors and margins computed programmatically via scikit-learn SVC (with analytical fallback)
- 🌟 **60 FPS Target** — BufferGeometry, shared materials, additive blending, no per-frame allocations
- 📐 **Math Overlay** — Current state, LaTeX-style formula, and educational explanation always visible
- 🖱 **OrbitControls** — Interactive 3D camera in KERNEL_3D state (rotate/zoom/pan)
- ♿ **Graceful Degradation** — Falls back if UnrealBloomPass or scikit-learn is unavailable

---

## 🔧 Local Development

```bash
# 1. Clone the repository
git clone https://github.com/miccowang66-max/VSM-3D-Math-animation.git
cd VSM-3D-Math-animation

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> Python 3.10+ is required.

---

## 🚀 Deployment

### Deploy to Streamlit Community Cloud (Free)

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select this repository
4. Set **Main file path** to `streamlit_app.py`
5. Click **"Deploy!"**

Streamlit Cloud auto-installs `requirements.txt` — zero config needed.

### Deploy to Hugging Face Spaces

1. Create a new **Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** SDK
3. Push this repo to the Space
4. The app auto-deploys

---

## 📁 Project Structure

```
VSM-3D-Math-animation/
├── streamlit_app.py          # Main application (1,050+ lines)
│   ├── generate_datasets()   # Python: linear/nonlinear data + SVM computation
│   ├── build_html()          # Python: generates complete Three.js embed
│   └── main()                # Streamlit entry point
├── requirements.txt           # Pinned Python dependencies
├── design.md                  # Project design specification
├── CLAUDE.md                  # AI coding assistant guidance
└── .gitignore                 # Python gitignore rules
```

### How It Works

1. **Python (backend):** Generates the dataset, runs scikit-learn SVM, computes decision boundary geometry, kernel lift z-values, and support vector indices.
2. **JSON bridge:** All computed data is serialized and injected into the HTML template.
3. **Three.js (frontend):** Embedded in Streamlit via `st.components.html()`. Manages the scene graph, animation loop, state machine transitions, OrbitControls, and post-processing — all in-browser with zero server round-trips after load.

---

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Python web app framework + component embedding |
| [Three.js r160](https://threejs.org) | WebGL 3D rendering engine (ES modules via importmap) |
| [scikit-learn](https://scikit-learn.org) | SVM training + support vector detection |
| [NumPy](https://numpy.org) | Dataset generation + numerical computation |
| [OrbitControls](https://threejs.org/docs/#examples/en/controls/OrbitControls) | Interactive 3D camera controls |
| [EffectComposer](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer) | Post-processing pipeline |
| [UnrealBloomPass](https://threejs.org/docs/#examples/en/postprocessing/UnrealBloomPass) | Glow/bloom effect |

---

## 📐 Math Background

### Linear SVM

Given training data \\((x_i, y_i)\\) with \\(y_i \\in \\{-1, +1\\}\\), find \\(w\\) and \\(b\\) that maximize the margin:

\\[
\\min_{w,b} \\frac{1}{2}\\|w\\|^2 \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\geq 1
\\]

- **Decision Boundary:** \\(w^T x + b = 0\\)
- **Margins:** \\(w^T x + b = \\pm 1\\)
- **Support Vectors:** Points lying exactly on the margins

### Kernel Trick

For nonlinearly separable data, map to a higher-dimensional feature space:

\\[
\\Phi(x_1, x_2) = (x_1, x_2, e^{-(x_1^2 + x_2^2)})
\\]

The kernel function \\(K(x, y) = \\langle\\Phi(x), \\Phi(y)\\rangle\\) computes the dot product in feature space without explicitly computing the coordinates:

\\[
K(x, y) = \\exp(-\\gamma\\|x - y\\|^2) \\quad \\text{(RBF Kernel)}
\\]

> ⚠️ **Note:** The z = exp(−(x²+y²)) mapping shown in this visualization is an intuitive approximation. The true RBF kernel maps to an **infinite-dimensional** feature space, not just 3D.

---

## 📄 License

MIT

## About

An educational 3D visualization of the SVM kernel trick, built with Three.js and Streamlit for the VSM Math Animation project.
