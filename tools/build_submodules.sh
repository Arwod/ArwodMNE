#!/bin/bash
# This script downloads and builds submodules for the mne-cpp project 
#
# This file is part of the MNE-CPP project. For more information visit: https://mne-cpp.github.io/
#
# This script is based on an open-source cross-platform script template.
# For more information you can visit: https://github.com/juangpc/multiplatform_bash_cmd
#

# ######################################################
# ########### LINUX/MAC SECTION ########################

SCRIPT_PATH="$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"
BASE_PATH=${SCRIPT_PATH}/..

argc=$#
argv=("$@")

for (( j=0; j<argc; j++)); do
    if [ "${argv[j]}" == "lsl" ]; then
        cd ${BASE_PATH}
        git submodule update --init src/applications/mne_scan/plugins/lsladapter/liblsl
        cd src/applications/mne_scan/plugins/lsladapter/liblsl
        mkdir build
        cd build
        cmake ..
        cmake --build .
    elif [ "${argv[j]}" == "brainflow" ]; then
        cd ${BASE_PATH}
        git submodule update --init src/applications/mne_scan/plugins/brainflowboard/brainflow
        cd src/applications/mne_scan/plugins/brainflowboard/brainflow
        mkdir build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX=../installed -DCMAKE_BUILD_TYPE=Release ..
        cmake --build .
    fi
done

# ########### LINUX/MAC SECTION ENDS ###################
# ######################################################

exit 0
