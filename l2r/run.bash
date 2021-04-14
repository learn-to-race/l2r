#!/bin/bash
# ========================================================================= #
# Filename:                                                                 #
#    run.bash                                                               #
#                                                                           #
# Description:                                                              # 
#    Convenient running script                                              #
# ========================================================================= #
export PYTHONPATH=${PWD}:$PYTHONPATH

print_usage()
{
	echo "Convenience script to load parameters and run an agent"
	echo "Usage (baseline model): ./run.bash -b <baseline>"
	echo "Usage (other model): ./run.bash -s <script> -c <config>"
	echo ""
	echo " Arguments:"
	echo "  -h, --help      |  shows this message"
	echo "  -b, --baseline  |  run a baseline model ['random', 'sac', 'mpc']"
	echo "  -c, --config    |  configuration file, passed as -c to script"
	echo "  -s, --script    |  python script to run"
}

while [[ $# -gt 0 ]]; do
  opt="$1"
  shift;
  current_arg="$1"
  case "$opt" in
    "-h"|"--help"       ) print_usage; exit 1;;
    "-b"|"--baseline"   ) BASELINE="$1"; shift;;
    "-c"|"--config"     ) CONFIG="$1"; shift;;
    "-s"|"--script"     ) SCRIPT="$1"; shift;;
    *                   ) echo "Invalid option: \""$opt"\"" >&2
                        exit 1;;
  esac
done

if [ "$BASELINE" == "" ] && [ "$SCRIPT" = "" ]; then
    echo "Invalid arguments. Please a baseline (with -b) or script (with -s)"
    exit 1;
fi

if [ "$BASELINE" != "" ]; then
    if [ "$BASELINE" == "random" ]; then
        python3 scripts/runner_random.py configs/params_random.yaml
        exit 0;
    elif [ "$BASELINE" == "sac" ]; then
        python3 scripts/runner_sac.py configs/params_sac.yaml
        exit 0;
    elif [ "$BASELINE" == "mpc" ]; then
        python3 scripts/runner_mpc.py configs/params_mpc.yaml
        exit 0;
    else
        echo "Expecting baseline to be in ['random', 'sac', 'mpc']"
        exit 1;
    fi
fi

if [ "$CONFIG" == "" ]; then
    echo "Please include a configuration file with -c or --config"
fi

COMMAND="${SCRIPT} -c ${CONFIG}"
python3 ${COMMAND}
