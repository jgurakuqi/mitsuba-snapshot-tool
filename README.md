# mitsuba-snapshot-tool


## Table of Contents

- [Description](#Description)
- [Installation](#installation)
- [Ply meshes](#ply-meshes)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description

The goal of this project is to develop a powerful and user-friendly tool that allows users to capture snapshots from various perspectives of a Mitsuba XML scene and facilitate the creation of more complex scenes through the combination of starting files.


## Ply meshes

- Unfortunately, some of the vripped meshes provided by [Stanford](https://graphics.stanford.edu/data/3Dscanrep/) have missing/brokwn normals, hence they are rendered with black pixels on mitsuba. I chose to exclude them from the repository. The problematic ones are:
    - Dragon.
    - Happy Budda.
    - Lucy (angel)

- Every used mesh was binarized through [Meshlab](https://www.meshlab.net/) to improve performance (as indicated by Mitsuba).

- Except for the Bunny mesh, every other mesh had to be rescaled in terms of dimensions, to be able to provide similar lightning
sources and camera positions to all the scenes (even if some tweaks are still needed due to different shapes). Also:
    - The armadillo was rotated to face the camera.
    - The thai statue was 

- Every processing operation applied over a ply file is reported into the filename:
    - Rot = Rotated to face the camera.
    - bin = binarized for producing scenes more efficiently.
    - reduced = compressed the number of faces to reduce the file size in case of very large mesh files.
    - resc = rescaled, i.e., all brought into a ~0.16 height, in order to be able to use closer cameras and smaller x/y/z transformations. 

- I rescaled all the ply mashes to make them of nearly the same size, in order to fit them in the xml Mitsuba scenes with the most similar possible environmental settings (i.e., light source position, object position and scale). Obviously, due to the different shapes, I was still forced to tweak a little the different settings to fit the object in the scene.

- I rotated:
    - Lucy by -90 wrt to the X axis and by -180 on the Y axis.
    - Armadillo 

## Installation

The requirements for this project are:
- [Python 3](https://www.python.org/downloads/)
- [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3)
- [Pytorch](https://pytorch.org/) for GPU memory deallocation if you choose to use CUDA. 
  
```bash
pip install numpy
pip install matplotlib
```

## Usage

To run the project can be used an IDE or the following python command:
```bash
python main.py
```

## Contributing

```bash
git clone https://github.com/jgurakuqi/mitsuba-snapshot-tool.git
```

## License

MIT License

Copyright (c) 2023 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
