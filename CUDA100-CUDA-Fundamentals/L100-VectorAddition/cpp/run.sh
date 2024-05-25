#!/usr/bin/env bash

./run-cpu.sh $@

./run-cpu-avx.sh $@

./run-cpu-multi-threads.sh $@

./run-gpu-cuda.sh $@