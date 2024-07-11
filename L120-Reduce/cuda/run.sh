#!/usr/bin/env bash

# Bitonic sort - currently only works with array size of multiple of 2
# ~128 million (multiple of 2) = 134217728
# ~500 million (multiple of 2) = 536870912
# ~1 billion (multiple of 2) = 1073741824 - This takes very long compare to ~500 million, why?

# ARRAY_SIZE=134217728
ARRAY_SIZE=536870912
# ARRAY_SIZE=1073741824

for op in min max avg median sort; do
  ./reduce.ex $ARRAY_SIZE $op
  echo
done
