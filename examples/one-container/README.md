## Overview

This configuration has the agent running in the same container as the Arrival simulator. 


## Build image

```bash
docker build -f Dockerfile -t l2r:latest .
```

## Run container, GPU required for the simulator
```bash
docker run -it --rm --gpus=all l2r:latest python3.9 main.py
```
