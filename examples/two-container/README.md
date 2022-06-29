## Overview

This configuration has the agent and l2r framework running in a separate container
than the Arrival simulator.

## Build image

```bash
docker build -f Dockerfile -t l2r:latest .
```

## Run containers

```bash
docker compose --env-file .env-vars up
```
