set -e

# install dependencies
sudo apt update
sudo apt install openjdk-17-jdk
sudo apt install python3-tk
sudo apt install cgroup-tools

# create cgroup for tests
sudo cgcreate -t $USER: -a $USER: -g memory:postgres_pbm