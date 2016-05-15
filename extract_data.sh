#!/bin/bash

for filename in snli/*.xz
do
    echo "extracting $filename";
    tar -xJvf $filename -C snli/
done
