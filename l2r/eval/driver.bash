#!/bin/bash

CONDA_FILE=environment.yml
PIP_FILE=requirements.txt
VENV_FILE=venv/bin/activate

print_usage()
{
	echo "Submission script which is invoked in the submission Docker"
	echo "container. Participants will not have access to this script, so"
	echo "modifications are not allowed. This script performs two primary"
	echo "operations:"
	echo ""
	echo "  1. Setup Python3 environment"
	echo ""
	echo "     1.1 For conda environments, provide 'environment.yml'"
	echo "     1.2 For virtual environments, provide the entire venv directory"
	echo "     1.3 For pip installations, provide requirements.txt"
	echo "         Note: this will be performed after the above steps"
	echo ""
	echo "  2. Runs the evalution script, eval.py"
	echo ""
	echo " Arguments:"
	echo "  -h, --help      |  shows this message"
	echo "  -c, --config    |  configuration file"
	echo "  -d, --duration  |  duration of pre-evaluationj, in seconds"
	echo "  -e, --episodes  |  number of evaluation episodes"
	echo "  -r, --racetrack |  evaluation racetrack name"
	echo ""
}

while [[ $# -gt 0 ]]; do
  opt="$1"
  shift;
  current_arg="$1"
  case "$opt" in
    "-h"|"--help"       ) print_usage; exit 1;;
    "-c"|"--config"     ) CONFIG="$1"; shift;;
    "-d"|"--duration"   ) DURATION="$1"; shift;;
    "-e"|"--episodes"   ) EPISODES="$1"; shift;;
    "-r"|"--racetrack"  ) RACETRACK="$1"; shift;;
    *                   ) echo "Invalid option: \""$opt"\"" >&2
                        exit 1;;
  esac
done

# Create and activate conda environment, if exists
if [ -f "$CONDA_FILE" ]; then
    echo "Found conda file"
    conda env create -f $CONDA_FILE -n venv
    conda activate venv

# Alternatively, if a virtual environment is provided, activate that
elif [ -f "$VENV_FILE" ]; then
    echo "Found virtual environment. Activating."
    source $VENV_FILE
fi

# Install pip requirements, if exists
if [ -f "$PIP_FILE" ]; then
    echo "Found requirements"
    pip3 install -r $PIP_FILE
fi

export PYTHONPATH=${PWD}:$PYTHONPATH
python3 eval.py --config $CONFIG --duration $DURATION --episodes $EPISODES \
        --racetrack $RACETRACK 
