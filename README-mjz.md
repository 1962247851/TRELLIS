### Ubuntu 22.04,RTX 4090, CUDA 11.8

export http_proxy='http://127.0.0.1:8889' export https_proxy='http://127.0.0.1:8889'


FileNotFoundError: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'
 
export CUDA_HOME=/usr/local/cuda-11.8