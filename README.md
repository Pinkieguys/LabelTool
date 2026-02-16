# LabelTool: Scientific Image Segmentation Refinement

⚠️ **Note:** This repository contains a **preliminary version** of the LabelTool toolkit.  
The code and data are provided for transparency and reproducibility, but the project is **still under active refinement**.  
Users may encounter incomplete documentation or minor issues. A more polished and fully documented release is currently under preparation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

LabelTool is a Python-based toolkit designed for scientific researchers to refine 3D image segmentations. It provides specialized tools to detect and fix common segmentation errors like over-segmentation (particles split into multiple labels) and under-segmentation (multiple particles merged into one label), which are frequent in X-ray CT and other 3D imaging modalities.

## Key Features

- **Interactive 3D GUI**: Built with PyQt5 and PyVista for real-time visualization and editing of 3D labels.
- **Automated Repair Algorithms**: Includes predefined workflows for fixing over- and under-segmentation.
- **Scientific Integration**: Seamlessly works with `spam` (Software for Processing and Analysis of Materials) and other scientific Python libraries.
- **High Performance**: Optimized 3D contour generation and contact analysis for large volumes.

## Project Structure

```text
labeltool/
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

## Usage

### Launching the GUI
After installation, you can launch the interactive tool directly:
```bash
labeltool
```
Or via python:
```bash
python -m labeltool
```

### Examples
Check the `examples/` directory for scripts demonstrating automated refinement:
- `python examples/fix_over_segmentation.py`: Simulate and repair over-segmented particles.
- `python examples/fix_under_segmentation.py`: Simulate and repair under-segmented particles.


### Segmentation Repair Scripts
To run the automated repair examples:
```bash
python fix_over_segmentation.py
python fix_under_segmentation.py
```