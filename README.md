# Self-Driving-Cars Final Project
## Steps for installing SEEM
After cloning the repo on ARC/Great Lakes, run the following commands:
```bash
cd /scratch/na565s001f24_class_root/na565s001f24_class/skwirskj/Segment-Everything-Everywhere-All-At-Once
module load python3.10-anaconda
conda create -n SEEM-env python==3.10 -y
conda activate SEEM-env
module load cuda/11.8.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge mpi4py==3.1.5 openmpi -y
```

Now following the install instructions from SEEM:
```bash
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
```

To delete the `conda` environment:
```bash
conda remove -n SEEM-env --all
```

## Steps for Loading a Pretrained SEEM Model

First download the model:
```bash
wget https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
```

Next, create the environment and install the deps:
```bash
module load python3.10-anaconda
conda create -n SEEM-env python==3.10 -y
conda activate SEEM-env
module load cuda/11.8.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Run the `load_seem.py` script to load the model and pass the path to the download:
```bash
python load_seem.py --model-path=<path/to/model>
```