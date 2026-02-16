# LabelTool: Scientific Image Segmentation Refinement

⚠️ **Note:** This repository contains a **preliminary version** of the LabelTool toolkit.  
The code and data are provided for transparency and reproducibility, but the project is **still under active refinement**.  
Users may encounter incomplete documentation or minor issues. A more polished and fully documented release is currently under preparation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

LabelTool is a Python-based toolkit designed for scientific researchers to refine 3D image segmentations. It provides specialized tools to detect and fix common segmentation errors like over-segmentation (particles split into multiple labels) and under-segmentation (multiple particles merged into one label), which are frequent in X-ray CT and other 3D imaging modalities.

This repository also includes specific implementations of algorithms discussed in our associated paper, including **Eigenvalue Correction** for normal fitting and **LLT (Locally-adaptive Local Threshold)** technique for contact correction.

## Key Features

- **Interactive 3D GUI**: Built with PyQt5 and PyVista for real-time visualization and editing of 3D labels.
- **Automated Repair Algorithms**: Includes predefined workflows for fixing over- and under-segmentation.
- **Scientific Integration**: Seamlessly works with `spam` (Software for Processing and Analysis of Materials) and other scientific Python libraries.
- **Advanced Contact Analysis**: Implements improved contact normal detection and segmentation correction algorithms.

## Project Structure

```text
labeltool/
├── algo/               # <--- Improved algorithms (SPAM patches) from the paper
│   ├── contacts.py     # Modified contact normal fitting (Eigenvalue Correction)
│   └── modifiedlabel_pool.py # Modified local detection (LLT technique)
├── dataset/            # <--- Experimental datasets used in the paper
│   ├── NaturalSand/    # Natural Sand samples (Raw image & Golden Standard)
│   └── Lentils/        # Lentils samples (Raw image & Golden Standard)
├── src/labeltool/      # Core package source code
│   ├── gui.py          # Interactive GUI implementation
│   ├── segmentation.py # Segmentation refinement algorithms
│   ├── viz_utils.py    # 3D visualization utilities
│   └── merge_utils.py  # Label merging and boundary detection
├── examples/           # Demonstration scripts and tutorials
├── tests/              # Unit tests
├── pyproject.toml      # Project metadata and dependencies
└── README.md
```

### Dependencies
The tool requires several scientific Python libraries, which will be installed automatically:
- `numpy`, `scipy`, `scikit-image`
- `PyQt5`, `pyqtgraph`
- `pyvista`
- `spam-package` (Software for Processing and Analysis of Materials)
- `tifffile`

---

## Algorithms & SPAM Integration (Paper Implementation)

The `algo/` folder contains two core algorithms described in our paper. These are built upon the open-source `spam` library. Since these improvements have not yet been merged into the official `spam` release, we provide them here as patches.

To reproduce the results presented in the paper, please replace the corresponding files in your installed `spam` library with the files provided in the `algo/` folder:

### 1. Eigenvalue Correction for Normal Fitting
*   **File:** `algo/contacts.py`
*   **Function:** `_contactNormals_new`
*   **Description:** This function replaces the original `_contactNormals`. It implements the **Eigenvalue Correction Algorithm** to provide more accurate normal vector fitting at contact points.
*   **Usage:** Locate `spam/kdtree/contacts.py` (or equivalent path in your installation) and replace it with `algo/contacts.py`.

### 2. LLT Technique for Contact Correction
*   **File:** `algo/modifiedlabel_pool.py`
*   **Function:** `localDetection_modiefied`
*   **Description:** This file is a replacement for `spam`'s pool processing module. The `localDetection_modiefied` function replaces `localDetection` to implement the **LLT** technique for precise contact correction and segmentation refinement.
*   **Usage:** Locate `pool.py` in the `spam` library and replace it with `algo/modifiedlabel_pool.py`.

> **Tip:** We recommend backing up the original `spam` files before overwriting them.

---

## Datasets

The `dataset/` directory contains the experimental data used to validate the algorithms. Each subfolder contains both the raw grayscale images and the Golden Standard (GT) labels.

*   **NaturalSand**:
    *   Contains tomographic scans of natural sand particles.
    *   Includes `Raw Image` and manually corrected `Golden Standard`.
*   **Lentils**:
    *   Contains tomographic scans of lentils.
    *   Includes `Raw Image` and manually corrected `Golden Standard`.
