#!/bin/bash
# This script runs home_surveillance as the current user rather than root
# but will rquire elevated privilegs to launch the container itself until
# rootless Docker or similar container runtime is ported to aarch64 with 
# nvidia runtime support

# image data is mounted in ~/.home_surveillance/

set -ex

readonly ALIGNED_IMAGES_DIR="$HOME/.home_surveillance/aligned-images"
readonly TRAINING_IMAGES_DIR="$HOME/.home_surveillance/training-images"
readonly USER_ID=$(id -u)
readonly GROUP_ID=$(cut -d: -f3 < <(getent group video))

setup () {
    mkdir -p $ALIGNED_IMAGES_DIR
    mkdir -p $TRAINING_IMAGES_DIR
}

run () {
    sudo docker run \
    -it --rm --runtime nvidia --user $USER_ID:$GROUP_ID \
    -v $ALIGNED_IMAGES_DIR:/var/home_surveillance/aligned-images \
    -v $TRAINING_IMAGES_DIR:/var/home_surveillance/training-images \
    -p 5000:5000 domcross/home_surveillance_jetson:latest
}

main () {
#    setup
    run
}

main
