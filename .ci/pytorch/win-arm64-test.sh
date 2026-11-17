#!/bin/bash
set -ex -o pipefail

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

run_tests() {
    if [[ "${SHARD_NUMBER}" == 1 ]]; then
        echo Running smoke_test.py...
        python ./.ci/pytorch/smoke_test/smoke_test.py --package torchonly
    fi

    echo Running Windows CPU Python tests...
    cd test

    # Keep selection aligned with win-test-helpers/test_python_shard.bat.
    python run_test.py --exclude-distributed-tests --exclude-quantization-tests \
        --shard "${SHARD_NUMBER}" "${NUM_TEST_SHARDS}" --verbose
}

run_tests
echo "TEST PASSED"
