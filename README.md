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
To run on arc, first ssh in and load `singularity 4.3.1` into the session, then run the container:
```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
# You do not need to do this step if you're not running anything with GPUs
salloc --account=na565s001f24_class --partition=gpu -c 16 --gpus=1 --mem=8GB --time=01:00:00
module load singularity/4.3.1
cd skwirskj/singularity_run
singularity shell --nv --overlay seem_overlay.img seem_container.sif
```

which will run whatever script you have placed under the `%runscript` section in [`seem.def`](./singularity/seem.def).

# Datasets
## Cityscapes install
Following the install steps, you have access to scripts that can assist with the setup of the cityscapes dataset. For more information, visit [cityscapes github](https://github.com/mcordts/cityscapesScripts). Note that only X-Decoder will be used for this dataset.

```bash
csDownload -d Segment-Everything-Everywhere-All-At-Once/.xdecoder_data/cityscapes gtFine_trainvaltest.zip
unzip gtFine_trainvaltest.zip
csDownload -d Segment-Everything-Everywhere-All-At-Once/.xdecoder_data/cityscapes leftImg8bit_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
csCreateTrainIdLabelImgs
csCreatePanopticImgs
```

The dataset is now prepared!

## Installation Guide for SAM
`pip install git+https://github.com/facebookresearch/segment-anything.git`

installing dependencies:
`pip install opencv-python pycocotools matplotlib onnxruntime onnx`