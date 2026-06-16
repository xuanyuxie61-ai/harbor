#!/bin/bash
set -u

mkdir -p /logs/verifier
cd /app/workspace || exit 1

REFERENCE_HASH="6b76951c0c41bac7025eb4f1e1a49c49a682c94c410560a5e050848cbd3f7944"

if [ -f ./executable ]; then
  CURRENT_HASH="$(sha256sum ./executable | awk '{print $1}')"
  if [ "$CURRENT_HASH" = "$REFERENCE_HASH" ]; then
    echo "Removing unchanged reference executable before build."
    rm -f ./executable
  fi
fi

BUILD_OK=0
if [ -f ./compile.sh ]; then
  chmod +x ./compile.sh
  ./compile.sh
  BUILD_OK=$?
elif [ -f ./build.sh ]; then
  chmod +x ./build.sh
  ./build.sh
  BUILD_OK=$?
else
  echo "No compile.sh or build.sh found in /app/workspace" >&2
  BUILD_OK=1
fi

if [ "$BUILD_OK" -ne 0 ] || [ ! -x ./executable ]; then
  echo '{"score":0.0,"passed":0,"total":30,"status":"fail","failures":["build_failed_or_no_executable"]}' > /logs/verifier/reward.json
  echo "0.0" > /logs/verifier/reward.txt
  exit 1
fi

PROGRAM_UNDER_TEST=/app/workspace/executable python -m pytest -q /tests/test_main.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo '{"score":1.0,"passed":30,"total":30,"status":"pass","failures":[]}' > /logs/verifier/reward.json
  echo "1.0" > /logs/verifier/reward.txt
else
  echo '{"score":0.0,"passed":0,"total":30,"status":"fail","failures":["test_failed"]}' > /logs/verifier/reward.json
  echo "0.0" > /logs/verifier/reward.txt
fi

exit $EXIT_CODE
