# mitsuba-snapshot-tool


## Table of Contents

- [Description](#Description)
- [Installation](#installation)
- [Ply meshes](#ply-meshes)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description

The goal of this project is to develop a powerful and user-friendly tool that allows users to produce a dataset of synthetic images for the purpose of testing Shape from Polarization methods. Currently the project supports multiple shapes, materials and Fov angles. Originally, it also supported rendering from different viewpoints, but this functionality has now been abandoned because this project was developed on a par with another aimed at evaluating the performance of the 4 Shape from Polarisation method.
In the future, the project will be expanded to again include the different viewpoints and possibly further configurations through the following modifications:
* Scenes with a standard light source and a linear polarising lens placed between the camera and the target object.
* Combinable scenes, allowing several scenes to be joined together to create more complex scenes, testing methods that can extract multiple shapes from a single scene.
* More shapes.
* More materials.
* More Fov angles (up to the 180deg limit of Mitsuba 3).
* More light sources and at different distances to simulate more realistic environments.




## Ply meshes

- Unfortunately, some of the vripped meshes provided by [Stanford](https://graphics.stanford.edu/data/3Dscanrep/) have missing/brokwn normals, hence they are rendered with black pixels on mitsuba. I chose to exclude them from the repository. The problematic ones are:
    - Dragon.
    - Happy Budda.
    - Lucy (angel)

- Every used mesh was binarized through [Meshlab](https://www.meshlab.net/) to improve performance (as indicated by Mitsuba).


- I rescaled all the ply mashes to make them of nearly the same size, in order to fit them in the xml Mitsuba scenes with the most similar possible environmental settings (i.e., light source position, object position and scale). Obviously, due to the different shapes, I was still forced to tweak a little the different settings to fit the object in the scene.

- Currently, none of the requested ply files are provided due to their large sizes, but they can be recovered here [Stanford](https://graphics.stanford.edu/data/3Dscanrep/) and placed in the related folder in the [scene_files folder](https://github.com/jgurakuqi/mitsuba-snapshot-tool/tree/main/scene_files). The file names in the xml scene files have to match the ply filenames. Later, the modified ply files and the pyramid file will be loaded.


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

## Mitsuba 3 citation

@software{jakob2022mitsuba3,
    title = {Mitsuba 3 renderer},
    author = {Wenzel Jakob and Sébastien Speierer and Nicolas Roussel and Merlin Nimier-David and Delio Vicini and Tizian Zeltner and Baptiste Nicolet and Miguel Crespo and Vincent Leroy and Ziyi Zhang},
    note = {https://mitsuba-renderer.org},
    version = {3.0.1},
    year = 2022,
}

## License

MIT License

Copyright (c) 2023 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
