#!/bin/bash
# ========================================================================= #
# Filename:                                                                 #
#    run.bash                                                               #
#                                                                           #
# Description:                                                              # 
#    Script to run the runner.py script                                     #
# ========================================================================= #
export PYTHONPATH=${PWD}:$PYTHONPATH

if [ "$1" = "random" ]; then
	python3 scripts/runner_random.py configs/params_random.yaml
elif [ "$1" = "sac" ]; then
	python3 scripts/runner_sac.py configs/params_sac.yaml
else
	python3 scripts/runner.py configs/params.yaml
fi