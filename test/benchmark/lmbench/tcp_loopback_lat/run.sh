#!/bin/sh

# SPDX-License-Identifier: MPL-2.0

set -e

echo "*** Running lmbench TCP latency test ***"

/benchmark/bin/lmbench/lat_tcp -s 127.0.0.1
/benchmark/bin/lmbench/lat_tcp -P 1 127.0.0.1
/benchmark/bin/lmbench/lat_tcp -S 127.0.0.1
