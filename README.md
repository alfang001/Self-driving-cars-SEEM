# Self-Driving-Cars Final Project
Clone this project using the following command so that the SEEM project is also cloned:
```bash
git clone --recursive git@github.com:alfang001/Self-driving-cars-SEEM.git
```

## Steps for installing SEEM
After cloning the repo on ARC/Great Lakes, run the following commands. Note that this has not been able to work yet:
```bash
cd /scratch/na565s001f24_class_root/na565s001f24_class/skwirskj/Segment-Everything-Everywhere-All-At-Once
module load python3.10-anaconda
conda create -n SEEM-env python==3.10 -y
conda activate SEEM-env
module load cuda/11.8.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Before running the pip install commands, you need to make sure that openmpi is installed. I have not yet tested this on ARC/Great Lakes, but this is the command you would use on Ubuntu:
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

Now following the install instructions from SEEM:
```bash
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../
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