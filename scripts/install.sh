source ~/anaconda3/etc/profile.d/conda.sh
conda activate guided-research-ag
git clone https://github.com/facebookresearch/higher.git
cd higher && pip install .
cd ..
git clone https://github.com/TUM-KDD/seml.git
cd seml && pip install .
