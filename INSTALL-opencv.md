## Redhat 9, Rocky 9 via DNF
```
sudo dnf config-manager --set-enabled crb
sudo dnf install opencv-devel

# Verify opencv installation
pkg-config --modversion opencv4
```

## Redhat 9, Rocky 9 via Source Code compilation
```
# Basic build packages
sudo dnf install cmake gcc gcc-c++ gtk2-devel pkgconfig
sudo dnf install python3 python3-devel numpy

# Pick the git directory for opencv repo
cd <OPENCV-GIT-DIRECTORY>

git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git
mkdir -p ./build
cd ./build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j

# system-wide installation
sudo make install

# Verify opencv installation
pkg-config --modversion opencv4

```