# Colorwheel Orientation

## Getting started

### Option 1: Run in Jupyter notebook

In a new Python enviroment, install the packages listed in docker/requirements.txt, as follows:

```
python -m venv env
source env/bin/activate
pip install -r docker/requirements.txt
```

Open a jupyter notebook:
```
cd mlex_colorwheel/src
jupyter notebook
```

In Jupyter, execute the file frontend.ipynb, and the app will be available at [localhost:8050](http://localhost:8050)

### Option 2: Run in Docker

Installing docker desktop:

  - [Docker](https://docs.docker.com/get-docker/)
 
For Linx, install Docker and [Docker Compose (version 3 above)](https://docs.docker.com/compose/install/) 

To run, execute the following lines of code in terminal:
```
cd mlex_colorwheel
docker-compose up --build
```
Once built, you can access the app at [localhost:8050](http://localhost:8050)

### Preprocessing
Images should only contain actual pixels. Any irrelevant information, such as labels, cptions, scalebar should be cropped.

## Copyright
MLExchange Copyright (c) 2021, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
