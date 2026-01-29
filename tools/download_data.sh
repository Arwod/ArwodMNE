#!/bin/bash
# This script performs generates and copies the necesary library dependencies for running qt-projects both for 
# dynamic and for staic builds. 
#
# This file is part of the MNE-CPP project. For more information visit: https://mne-cpp.github.io/
#
# This script is based on an open-source cross-platform script template.
# For more information you can visit: https://github.com/juangpc/multiplatform_bash_cmd
#

function cleanAbsPath()
{
    local  cleanAbsPathStr="$( #spawns a new bash interpreter
        cd "$1" >/dev/null 2>&1 #change directory to that folder
        pwd -P
    )"
    echo "$cleanAbsPathStr"
}

argc=$#
argv=("$@")
EXIT_SUCCESS=0
EXIT_FAIL=1
ScriptPath="$(cleanAbsPath "$(dirname "$0")")"
BasePath="$(cleanAbsPath "$ScriptPath/..")"
ResourcesPath="$(cleanAbsPath "$BasePath/resources")"
DownloadTestData="false" 
DownloadSampleData="false" 

doPrintConfiguration() {
  echo " "
  echo ========================================================================
  echo ======================== MNE-CPP DOWNLOAD DATA =========================
  echo " "
  echo " ScriptPath = $ScriptPath"
  echo " BasePath   = $BasePath"
  echo " ResourcesPath = $ResourcesPath"
  echo " "
  echo " DownloadTestData = $DownloadTestData"
  echo " DownloadSampleData  = $DownloadSampleData"
  echo " "
  echo ========================================================================
  echo ========================================================================
  echo " "
}

doPrintHelp() {
  echo " "
  echo "Download sample or test datasets into the resources/data folder. "
  echo "Usage: ./deploy.bat [Options]"
  echo " "
  echo "All options can be used in undefined order."
  echo " "
  echo "[help] - Print this help."
  echo "[all]  - Download all datasets."
  echo "[test-data]   - Download mne-cpp-test-data set."
  echo "[sample-data] - Download MNE Sample Data set."
  echo " "
}

#parse input args
if [ $argc -eq 0 ]; then
  PrintHelp="true"
fi

for (( j=0; j<argc; j++ )); do
    if [ "${argv[j]}" == "all" ]; then
        DownloadTestData="true"
        DownloadSampleData="true"
    elif [ "${argv[j]}" == "test-data" ]; then
        DownloadTestData="true"
    elif [ "${argv[j]}" == "sample-data" ]; then
        DownloadSampleData="true"
    elif [ "${argv[j]}" == "help" ]; then
        PrintHelp="true"
    fi
done

#execution starts here
if [ "${PrintHelp}" == "true" ]; then
    doPrintHelp
    exit ${EXIT_SUCCESS} 
fi

doPrintConfiguration

if [[ ${DownloadTestData} == "true" ]]; then
  git clone https://github.com/mne-tools/mne-cpp-test-data.git ${ResourcesPath}/data/mne-cpp-test-data
fi
if [[ ${DownloadSampleData} == "true" ]]; then
  curl -C - -o MNE-sample-data.tar.gz https://files.osf.io/v1/resources/rxvq7/providers/osfstorage/59c0e26f9ad5a1025c4ab159\?action\=download\&direct\&version\=6
  mv MNE-sample-data.tar.gz ${ResourcesPath}/data/
  currentFolder=`pwd`
  cd ${ResourcesPath}/data/
  tar -xvf MNE-sample-data.tar.gz 
  rm -fr MNE-sample-data.tar.gz
  cd ${currentFolder}
fi

exit 0
