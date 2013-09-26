#!/bin/bash
# This script installs liblbfgs on the current system

LIBLBFGS_URL="https://dl.dropboxusercontent.com/u/3091691/NNforMLL-lib/liblbfgs-1.10.tar.gz"

# this library will be put in the ./lib directory
rm -rf lib
mkdir -p lib
cd lib
curl $LIBLBFGS_URL -o liblbfgs-1.10.tar.gz

# Extract and install
tar xvf liblbfgs-1.10.tar.gz
rm liblbfgs-1.10.tar.gz
cd liblbfgs-1.10
./configure

# Admin password will be needed for these steps because we are installing to
# /usr/local/lib
echo "Going to install the library. You might need to enter an Admin password"
echo "because the library will be installed at a system path: /usr/local/lib"
sudo make
sudo make install


echo "Please check the output above to see if the library is installed at"
echo "/usr/local/lib. If installed at a different location, please update the"
echo "Makefile in this directory with the right path, for LIBLBFGS_PATH"
