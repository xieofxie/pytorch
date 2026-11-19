#!/bin/bash
set -ex -o pipefail

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

export SCRIPT_HELPERS_DIR="$SCRIPT_PARENT_DIR/win-arm64-test-helpers"

run_tests() {
    "$SCRIPT_HELPERS_DIR"/test_python_shard.bat
}

run_tests
echo "TEST PASSED"
