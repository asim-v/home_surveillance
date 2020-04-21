set -ex

readonly VERSION="0.1.0"

docker build --rm -t domcross/home_surveillance_jetson:latest .
docker tag domcross/home_surveillance_jetson:latest domcross/home_surveillance_jetson:$VERSION
