#!/bin/bash
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load apptainer/1.3.4
module load root/6.32.06
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
