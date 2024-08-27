<div align="center">

# SenNet + HOA - Hacking the Human Vasculature in 3D solution
https://www.kaggle.com/competitions/blood-vessel-segmentation/overview


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

Below you can find a outline of how to reproduce my solution for the SenNet + HOA - Hacking the Human Vasculature in 3D competition.
If you run into any trouble with the setup/code or have any questions please contact me at reachingxforward@gmail.com

###ARCHIVE CONTENTS
logs          : contains the trained models checkpoints and logs
train.sh : runs the code to train the models using the competition data to
train_pseudo_v2.sh : runs the code to train 2d model with pseudo labels
train_pseudo_3d.sh : runs the code to train 3d model with pseudo labels
src : source code for the trainings
configs: configs
data: contains the csv with path mappings for the training data.

###HARDWARE: (The following specs were used to create the original solution)
Ubuntu 22.04 LTS (1 TB NVME boot disk + 1 TB NVME cache)
TRX 2979X (24 CPUs, 128 GB of memory)
2 x NVIDIA RTX 4090

###SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10.9
CUDA 12.3
nvidia drivers v.545.23.08
Miniconda

###CODE SETUP
from git
```bash
# OPTION: 1 
# clone project
git clone https://github.com/burnmyletters/blood-vessel-segmentation-public
cd blood-vessel-segmentation-public

# OPTION: 2
# using the provided code
cd blood-vessel-segmentation-public

# create conda environment
conda create -n bvs python=3.10
conda activate bvs
conda install pip

# install requirements
pip install -r requirements.txt
```

###DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Below are the shell commands used in each step, as run from the top level directory (path_to_data)
```bash
pip install dtrx
cd data 
kaggle competitions download -c blood-vessel-segmentation
dtrx blood-vessel-segmentation.zip
```

Download the external dataset from the [Human Organ Atlas](http://human-organ-atlas.esrf.eu) site. If you want to skip the initial training without pseudo, you can download the images with pseudo-labels from Kaggle.

```bash
cd blood-vessel-segmentation
mkdir train_ext
cd train_ext
kaggle datasets download -d igorkrashenyi/50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2
dtrx 50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2.zip
rm 50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2.zip
mv 50um-ladaf-2020-31-kidney-pag-0-01-0-02-jp2 50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_ # needed to train the 2d model
```


###DATA PROCESSING
Before training the model you will need to generate additional projections for the data. For this you need to run
```bash
cd ../../
python generate_mutliview.py $(python parse_settings_json.py RAW_DATA_DIR)
```

###MODEL TRAINING

**NOTE:** assuming the SETTINGS.json is located in the code dir
**NOTE:** running the training code will create additional folders alongside with the original checkpoints  

To produce the full solution three stages are used. The first one (1) trains a model on the competition data and produce a pseudo labels on the external dataset . You can use provided checkpoint to inference the models and generate the pseudo labels, or you can use the labels provided from the previous section (generated in the same way).

The second stage (2) is separate training of 2d and 3d models on the extended dataset. The third stage (3) is an inference to reproduce the solution.

1) Stage 1 
```bash
sh ./train.sh
```

This run will generate initial checkpoints to create the pseudo labels. The checkpoints will be located in the logs/train/runs/ folder. 

**NOTE**: These checkpoints are also provided in the archive in logs/train/runs folder.

After the training is done run the following code using the trained model checkpoints or use provided checkpoints from the archive to generate pseudo:

```bash
python generate_pseudo.py --dataset_folder $(python parse_settings_json.py RAW_DATA_DIR) --logs_base_path $(python parse_settings_json.py LOGS_DIR)
```

**NOTE**: The pseudo labels are also provided in the 50um_LADAF-2020-31_kidney_pag-0.01_0.02_jp2_ folder.

2) Stage 2

You will need to perform the projection generation for the pseudo labels. For this run:
```bash
python generate_mutliview_pseudo.py $(python parse_settings_json.py RAW_DATA_DIR)
sh ./train_pseudo_v2.sh
sh ./train_pseudo_3d.sh
```

3) Inference
the inference code can be found https://www.kaggle.com/code/igorkrashenyi/4th-place-solution 
