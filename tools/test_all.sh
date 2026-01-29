#!/bin/bash

##### function definitions

function cleanAbsPath()
{
    local  cleanAbsPathStr="$( #spawns a new bash interpreter
        cd "$1" >/dev/null 2>&1 #change directory to that folder
        pwd -P
    )"
    echo "$cleanAbsPathStr"
}

doPrintConfiguration() {
  echo " "
  echo =========================================
  echo " VerboseMode = $VerboseMode"
  echo " RunCodeCoverage = $RunCodeCoverage"
  echo " BuildName = $BuildName"
  echo =========================================
  echo " "
}

doPrintHelp() {
  echo "Usage: ./test_all.bat [Options]"
  echo " "
  echo "All options can be used in undefined order."
  echo " "
  echo "[help] - Print this help."
  echo "[verbose] - Print tests output to in terminal."
  echo "[build-name=] - Specify the build-name of which to run its tests."
  echo " "
}

## input arguments parsing

argc=$#
argv=("$@")

VerboseMode="false"
RunCodeCoverage="false"
BuildName="Release"
PrintHelp="false"

for (( j=0; j<argc; j++)); do
  if [ "${argv[j]}" == "verbose" ]; then
    BUILD_COMMAND=1
    VerboseMode="true"
  elif [ "${argv[j]}" == "help" ]; then
    PrintHelp="true"
  elif [ "${argv[j]}" == "withCoverage" ]; then
    RunCodeCoverage="true"
  fi
  IFS='=' read -r -a inkarg <<< "${argv[j]}"
  if [ "${inkarg[0]}" == "build-name" ]; then
      BuildName="${inkarg[1]}"
  fi
done

if [ "${PrintHelp}" == "true" ]; then
    doPrintHelp
    exit ${EXIT_SUCCESS} 
fi

doPrintConfiguration

ScriptPath="$(cleanAbsPath "$(dirname "$0")")"
BasePath="$(cleanAbsPath "$ScriptPath/..")"

if [[ $(uname) == "Linux" ]]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BasePath/lib
fi

# start calling each test with some formatting
testColumnWidth=60
printf "%${testColumnWidth}s %s\n" " Test Name " " Result "

CompoundOutput=0
for test in $BasePath/out/${BuildName}/tests/test_*;
do
  # Run all tests and call gcov on all cpp files after each test run. Then upload to codecov for every test run.
  # Codecov is able to process multiple uploads and merge them as soon as the CI job is done.
  if [ "$VerboseMode" == "false" ]; then
    $test &> /dev/null
  else
    $test 
  fi
  lastReturnValue=$?

  if [ $lastReturnValue -ne 0 ]; then 
    CompoundOutput=$((CompoundOutput + 1))
    printf "%${testColumnWidth}s \e[91m\033[1m %s \033[0m\e[0m\n" "${test}" "Failed!"
    if [ "$ExitOnFirstFail" == "true" ];
    then
      exit $lastReturnValue
    fi
  else
    # echo ">> Test $test \t\t\t\t RockSolid!"
    printf "%${testColumnWidth}s \e[92m %s \e[0m\n" "${test}" "Rock Solid!"
  fi

  if [ "$RunCodeCoverage" == "true" ]; then
    find ./src/libraries -type f -name "*.cpp" -exec gcov {} \; &> /dev/null
    # Hide codecov output since it corrupts the log too much
    ./codecov &> /dev/null
  fi
done

exit $CompoundOutput
