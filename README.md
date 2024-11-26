# Self-Driving-Cars Final Project
## Steps for installing SEEM
After cloning the repo on ARC/Great Lakes, run the following commands:
```bash
module load python3.10-anaconda
conda create -n SEEM-env python==3.10 -y
conda activate SEEM-env
conda install -c conda-forge mpi4py==3.1.5 openmpi -y
```

Now following the install instructions from SEEM:
```bash
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
```