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
module load cuda/12.1.0
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
module load cuda/12.1.0
```

Run the `load_seem.py` script to load the model and pass the path to the download:
```bash
python load_seem.py --model-path=<path/to/model>
```


## Installation Guide for SAM
`pip install git+https://github.com/facebookresearch/segment-anything.git`

installing dependencies:
`pip install opencv-python pycocotools matplotlib onnxruntime onnx`

## Using Singularity Container on ARC
### Building the Container
Singularity allows us to install dependencies that would otherwise require `sudo`, which is not allowed on ARC. I used [lightning.ai](lightning.ai) to run the build steps. Two files need to be added to the lightning session from our repo, [`create_seem_singularity.sh`](./singularity/create_seem_singularity.sh) and [`seem.def`](./singularity/seem.def).

Once on lightning, you can run:

```bash
chmod +x create_seem_singularity.sh
sudo ./create_seem_singularity.sh
```

which will create a file called `seem_container.sif`. This is what is used on ARC to run the model within. To copy the container to ARC, use the following command
```bash
rsync -av -e ssh original/ uniqname@greatlakes.arc-ts.umich.edu:/path/to/destination/
```

### Running on ARC
To run on arc, first load `singularity 4.3.1` into the session, then run the container:
```bash
module load singularity/4.3.1
singularity run --nv vln_diffusion.sif
```

which will run whatever script you have placed under the `%runscript` section in [`seem.def`](./singularity/seem.def).