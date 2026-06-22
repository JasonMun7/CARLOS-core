# CARLOS-core

Continuous-time Adaptive Reinforcement Learning for Optimal Stopping.

Reference: [arxiv:2606.17545](https://arxiv.org/pdf/2606.17545)

## Build

Requires CMake and a C++20 compiler. On macOS, `brew install libomp` is optional — the extension uses
`std::thread` on macOS (to avoid libomp conflicts with PyTorch) and OpenMP on Linux.

```bash
pip install -r requirements.txt
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix .
```

## Test C++ bridge

```bash
python test_bridge.py
```

## Train v1 agent (B1 put smoke test)

```bash
PYTHONPATH=.:build python -m carlos.agent
```
