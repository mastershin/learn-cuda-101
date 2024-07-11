#!/usr/bin/env bash
ARRAY_SIZE=500m
for op in min max avg median sort; do
./reduce.ex $ARRAY_SIZE $op
done
