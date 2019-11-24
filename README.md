# Gaussian Processes and NTKs

Author: [Varun Sundar](github.com/varun19299)

![NNGP_NTK.gif](NNGP_NTK.gif)

## Install

### JAX

```bash
# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp27, cp35, cp36, cp37
CUDA_VERSION=cuda101  # alternatives: cuda90, cuda92, cuda100, cuda101
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.32-$PYTHON_VERSION-none-$PLATFORM.whl

pip install --upgrade jax  # install jax

```

### Neural Tangets

```bash
git clone https://github.com/google/neural-tangents
pip install -e neural-tangents
```

### Streamlit and Other Requirements

```\
pip install -r requirements.txt
```

## Running the App

`streamlit run GP_kernel.py` and follow the link provided.
