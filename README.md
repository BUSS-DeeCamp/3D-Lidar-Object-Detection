# 3D-Lidar-Object-Detection

author: phosphensvision

## Part 1 requirements

reference: <https://blog.csdn.net/Small_Munich/article/details/99053921>

in this section, 9 related packages will be installed: numpy, torch, numba, tensorboardX, easydict, pyyaml, scikit-image, tqdm and spconv.

the first 8 packages can be easily installed just follow one simple command, but the last package spconv is much more sophisticated to install and I present the instruction.

1. install numpy: `conda install numpy`

2. install torch>=1.1:  already installed

3. install numba: `conda install numba`

4. install tensorboardX: `conda install tensorboardX`

   <https://github.com/lanpa/tensorboardX> 

5. install easydict: `conda install easydict`

6. install pyyaml: `conda install pyyaml`

7. install scikit-image: `conda install scikit-image`

8. install tqdm: `conda activate tqdm`

next, how to install spconv:

you can refer to the tutorial: <https://github.com/traveller59/spconv>

or follow my tutorial: note that I install an old version of spconv, so please do not go to <https://github.com/traveller59/spconv> directly, go to [commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) to install spconv v1.0 instead.

in conda environment

1. git clone the repository:

   ```
   git clone https://github.com/traveller59/spconv.git --recursive #git clone the repository
   git checkout 8da6f967fb9a054d8870c3515b1b44eca2103634 #check out vesion 1.0
   ```

   

2. install boost:

   ```
   sudo apt-get install libboost-all-dev 
   ```

   if you encounter the error like this:

   ```
   E: 有几个软件包无法下载，要不运行 apt-get update 或者加上 --fix-missing 的选项再试试？
   ```

   and then you run the recommended command, but it doesn't help

   please refer to the website: <https://blog.csdn.net/qq_30354455/article/details/72739537>

3. download cmake >= 3.13.2 and make install: 

   first, you should search the website <https://cmake.org/files/v3.13/> and download the file [
   cmake-3.13.2.tar.gz](https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz)

   and then you enter the folder where the file is.

   ```
   tar xvf cmake-3.13.2.tar.gz
   cd  cmake-3.13.2
   ./configure
   make
   sudo make install 
   ```

   then you can check if cmake is successfully installed using the following command:

   ```
   cmake --version #and the info printed should becmake version 3.13.2
   ```

   

4. add cmake executables to PATH: 

   ```
   export  PATH=$PATH:/your_cmake_file_parent_path/cmake-3.13.2/bin
   source ~/.bashrc
   ```

   

5. setup spconv: 

   enter your spconv folder and run: 

   ```
   python  setup.py  bdist_wheel
   ```

   

6. the last step, install whl file:

   enter your dist list and you can find a spconv-xxx.whl file, run: 

   ```
   pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl
   ```





## Part 2 Data Process

in pytorch framework, there are mainly three class related to data preprocess:

1. torch.utils.data.Dataset
2. torch.utils.data.DataLoader
3. torch.utils.data.DataLoaderIter



### Dataset

there are 2 vital attribute of Dataset class

```
from torch.utils.data import DataLoader

# how to get the image from the raw data
def __getitem__(self, index):
        img_path, label = self.data[index].img_path, self.data[index].label
        img = Image.open(img_path)

        return img, label
        
# get the length of the dataset
def __len__(self):
        return len(self.data)
```

NOW, we build the BussDataset

### Some Details about the Dataset

1. the data filer: 

   ```
   ROOT_PATH
   ├── data
   │   ├── train_val_filter
   │   │   │── bin files
   │   ├── test_filter
   │   │   │── bin files
   │   ├── test_video_filter
   │   │   │── bin files
   │   ├── label_filter
   │   │   │── train_filter.txt
   │   │   │── val_filter.txt
   ```

   

2. the class index related to the class name: 

   | class_name | class_id |
   | ---------- | -------- |
   | Car        | 0        |
   | Truck      | 1        |
   | Tricar     | 2        |
   | Cyclist    | 3        |
   | Pedestrain | 4        |
   | Dontcare   | 5        |

   







