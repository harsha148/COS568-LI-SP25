#! /usr/bin/env bash
# set -e

# echo "Compiling benchmark..."
# git submodule update --init --recursive 

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8 

function generate_uint64_100M() {
    echo "Generating operations for $1"
    ./generate ../data/$1 2000000 --negative-lookup-ratio 0.5 # generate the lookup workload
    ./generate ../data/$1 2000000 --insert-ratio 0.5 --negative-lookup-ratio 0.5 # generate insert then lookup workload
    ./generate ../data/$1 2000000 --insert-ratio 0.9 --negative-lookup-ratio 0.5 --mix
    ./generate ../data/$1 2000000 --insert-ratio 0.1 --negative-lookup-ratio 0.5 --mix

}


for dataset in fb_100M_public_uint64; do
    generate_uint64_100M $dataset
done