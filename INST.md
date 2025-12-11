```shell
git clone https://github.com/stdcall0/LightweightMR
cd LightweightMR
conda create -n LMR python=3.9
conda activate LMR

pip install torch
pip install pymcubes trimesh open3d fpsample cython pyhocon timm
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install spconv-cu126

conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
MAX_JOBS=4 pip install flash-attn --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# Ensure lower version of numpy
pip install numpy==1.22.4

# The following line should only be ran once for each Featurize instance
sudo sed -i.bak -e 's/http:\/\/.*archive.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' -e 's/http:\/\/.*security.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && sudo apt update

# After the installation, you may need to reenter the conda env: conda activate LMR
sudo apt install libcgal-dev libboost-all-dev libgmp-dev libopen3d-dev cmake make binutils

# Build KDtree
cd models/cpplib
CC=gcc CXX=gcc python setup.py build_ext --inplace

# Build Delaunay (Exit conda env when building)
conda deactivate
cd ../../; cd models/delaunay_meshing/create_delaunay
cmake . -DOpen3D_DIR=/usr/lib/x86_64-linux-gnu/cmake/Open3D
make
cd ../create_mesh
cmake .
make
cd ../../..
conda activate LMR

# exec once (let open3d use system libstdc++)
#conda install -c conda-forge libstdcxx-ng (this not works for some reason)
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /environment/miniconda3/lib/libstdc++.so.6

# Run eval (with optimizations)
source scripts/run_sdf.sh
source scripts/run_vg.sh

# Run eval (without optimizations)
# remember to delete generated things
OPT_PERF=off source scripts/run_sdf.sh
OPT_PERF=off source scripts/run_vg.sh
```
