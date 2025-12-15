export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
# python setup.py build_ext --inplace
pip install -e .