# vLLM development setup recipe
setup-vllm name="vllm" arch="89-real":
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "Setting up vLLM in {{ name }}..."
    
    git clone https://github.com/vllm-project/vllm.git {{ name }}
    cd {{ name }}
    
    echo "Adding neuralmagic remote..."
    git remote add nm https://github.com/neuralmagic/vllm
    
    echo "Creating virtual environment..."
    uv venv --python=3.12
    source .venv/bin/activate
    
    echo "Installing build requirements..."
    uv pip install -r requirements/build.txt
    
    echo "Installing vLLM in editable mode..."
    VLLM_DISABLE_SCCACHE=1 CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v
    
    echo "Installing EP kernels with TORCH_CUDA_ARCH_LIST={{ arch }}..."
    pushd tools/ep_kernels
    TORCH_CUDA_ARCH_LIST="{{ arch }}" PIP_CMD="uv pip" bash install_python_libraries.sh
    popd
    
    echo "✅ vLLM setup complete in {{ name }}/"
