# Soft Robot Optimal Control via Reduced-Order Finite Element Models

This repository accompanies the paper "[Soft Robot Optimal Control via Reduced-Order Finite Element Models](https://arxiv.org/abs/2011.02092)" by Sander Tonkens, Joseph Lorenzetti, and Marco Pavone.

We provide an implementation of various optimal control algorithms (SCP, iLQR, LQR, etc.) for nonlinear reduced order
 models based on nonlinear Finite Element Models (FEMs). Specifically, it interfaces with the 
 [SOFA](https://www.sofa-framework.org/download/) simulation engine to simulate Soft Robots modeled using FEMs. 

---

## Dependencies

[SOFA](https://www.sofa-framework.org/download/): Sofa-framework is a physics simulation engine

SOFA plugins:

- [SOFA Python3](https://github.com/sofa-framework/SofaPython3): Python 3 development for python interface to SOFA. Installation help for out-of-tree builds can be found [here](https://github.com/SofaDefrost/plugin.SofaPython3/issues/137). Requires py37+
- [SOFA SoftRobots](https://github.com/SofaDefrost/SoftRobots)
- [SOFA STLIB](https://github.com/SofaDefrost/STLIB)

[ROS2](https://index.ros.org/doc/ros2/Installation/Foxy/): For using Sequential Convex Programming through cvxpy. See installation + setup below

Python packages: [cvxpy](https://www.cvxpy.org/install/), [control](https://python-control.readthedocs.io/en/0.8.3/intro.html#installation), [pyDOE](https://pythonhosted.org/pyDOE/) default numeric packages (numpy, scipy, matplotlib, etc.)

---

---


## Using the repository

Three models (Finger, Trunk and Diamond), copied from [SOFA SoftRobots](https://github.com/SofaDefrost/SoftRobots) plugin are provided.


The [examples folder](https://github.com/stonkens/trajopt_nlmor/tree/master/examples) is a great starting point once you have gone through the installation (see below).



---

---

## Notes

This project is under active development with limited documentation (for now). Any suggestions for improvements are welcome. 

We would like to acknowledge the great work [SOFA](https://www.sofa-framework.org/) does and encourage you to check it out and get involved.

---

---

## Install and setup environment

Tested platforms: Ubuntu: 20.04, Python 3.8 (tested with 3.8.2 and 3.8.3)

---

##### Get Plugin Libraries

Download the libraries SoftRobots and STLIB
```
mkdir $HOME/sofa-plugins
git clone https://github.com/SofaDefrost/SoftRobots $HOME/sofa-plugins/SoftRobots
git clone https://github.com/SofaDefrost/STLIB.git $HOME/sofa-plugins/STLIB
```
Now create a CMakeLists.txt file in the `$HOME/sofa-plugins/` directory and add the following:
```
cmake_minimum_required(VERSION 3.11)

find_package(SofaFramework)

sofa_add_plugin(STLIB/  STLIB VERSION 1.0)
sofa_add_plugin(SoftRobots/  SoftRobots VERSION 1.0)
```

---


##### Install Sofa

Follow [instructions](https://www.sofa-framework.org/community/doc/getting-started/build/linux/) to download dependencies (compiler, CMake, Qt, etc.)

Set up directory structure for Sofa and clone repository:
```
mkdir $HOME/sofa/src
mkdir $HOME/sofa/build
git clone https://github.com/sofa-framework/sofa.git $HOME/sofa/src
```

---

##### Build Sofa

Run `cmake-gui` to run the CMake GUI. Set the source path to `$HOME/sofa/src` and the build path to `$HOME/sofa/build`. Make sure the path to installation of Qt is correct by adding an entry `CMAKE_PREFIX_PATH` and setting it to the appropriate location (i.e. `/home/jlorenze/Qt/5.15.0/gcc_64`).

Run **Configure**, and set the compiler according to the instructions [here](https://www.sofa-framework.org/community/doc/getting-started/build/linux/).

Then, add entry `SOFA_BUILD_METIS` and enable it. Find the entry `SOFA_EXTERNAL_DIRECTORIES` and set it to `$HOME/sofa-plugins` where `$HOME` is replaced with the actual path (i.e. `/home/jlorenze/`). Also, add and enable entry `SOFTROBOTS_IGNORE_ERRORS` which will allow SoftRobots to compile without the STLIB library. Run **Configure** again (should complete with no errors), and then run **Generate**.

To build (use `-j` flag to use all cores):
```
cd $HOME/sofa/build
make -j
make install
```
Additionally, add `SoftRobots 1.0` to the file `$HOME/sofa/build/lib/plugin_ist.conf.default`.

---

##### Set up virtual environment (optional)

miniconda2 and anaconda3 (probably others, but these have been tested) can be used. SofaPython3 requires Python 3.8+. If using a package manager / virtual environment, ensure that `qmake --version` points to the same Qt version as Sofa (in order to enable running Sofa from the terminal (without `runSofa`))

```
conda create --name sofa python=3.8
conda activate sofa
```

Note that you can activate and deactivate this environment with ``conda activate`` and ``conda deactivate``.

---

##### Install python packages
```
conda install numpy, scipy
conda install pyqt
pip install slycot   # required for control package
pip install control
pip install pyDOE
pip install cvxpy
pip install osqp
pip install sip
```
Install pybind11 2.6.0 - DO NOT install 2.8.0 as it will throw compilation errors when compiling SofaPython3
```
conda install -c conda-forge pybind11=2.6.0
```
 There should now be a directory {PYTHON_ENV}/share/cmake/pybind11 (e.g. ${HOME}/anaconda3/envs/sofa/share/cmake/pybind11.
 This stores the cmake files for pybind11, necessary to compile SofaPython3. Track pybind2.8 issue here: [pybindissue](https://github.com/sofa-framework/SofaPython3/pull/216)

---
##### Install Gurobi

Install [Gurobi](https://cdn.gurobi.com/wp-content/plugins/hd_documentations/documentation/9.0/quickstart_linux.pdf)
```
tar xvfz ~/Downloads/gurobi9.5.0_linux64.tar.gz /opt/
cd /opt/
chmod +777 /opt/gurobi950
chmod +777 /opt/gurobi950/linux64
```
Once gurobi is installed, you need to install the license. Follow link above for instructions to get and install academic license. 
You then need to put the license in /opt/gurobi950 during `grbgetkey`.

Next, install Gurobi modules for Python (see [pyGurobi](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-))
```
cd $GUROBI_HOME
python setup.py install
```

Set your environment variables in `$HOME/.bashrc`. Should look like below
```
export PYTHONPATH="$HOME/sofa-plugins/STLIB/python3/src"
export SP3_BLD=$HOME/sofa-plugins/SofaPython3/build
export SOFA_BLD=$HOME/sofa/build
export GUROBI_HOME="/opt/gurobi950/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE=/opt/gurobi950/gurobi.lic
```
---
# TODO: Potentially get rid of this
##### Build SofaPython3

Now we can build:
```
cd $HOME/sofa-plugins/SofaPython3
mkdir build
```
Run `cmake-gui` and set the source path to `$HOME/sofa-plugins/SofaPython3` and bould path to `$HOME/sofa-plugins/SofaPython3/build`. Add entry `CMAKE_PREFIX_PATH` and set it to `$HOME/sofa/build/install;/path/to/Qt/version/gcc_64` where `$HOME` is the user's home directory (i.e. `/home/jlorenze/sofa/build/install;/home/jlorenze/Qt/5.15.0/gcc_64`). Also add entry `pybind11_DIR` which is the path to pybind11 version >= 2.3 (i.e. `/home/jlorenze/miniconda2/envs/sofa/share/cmake/pybind11` using Miniconda2 with environment named `sofa`). Run **Configure** and **Generate**.

Now build:
```
cd $HOME/sofa-plugins/SofaPython3/build
make -j
```

Now a couple of additional steps:
```
mkdir -p $(python3 -m site --user-site)
echo "export SP3_BLD=$HOME/sofa-plugins/SofaPython3/build" >> ~/.bashrc
echo "export SOFA_BLD=$HOME/sofa/build" >> ~/.bashrc
echo "export PYTHONPATH=$HOME/sofa-plugins/STLIB/python3/src"
source ~/.bashrc
ln -sFfv $(find $SP3_BLD/lib/python3/site-packages -maxdepth 1 -mindepth 1 -not -name "*.py") $(python3 -m site --user-site)
```
Adding the variables `SP3_BLD` and `SOFA_BLD` to the environment variables is a nice shortcut. Adding them to the `~/.bashrc` means they will automatically be defined on terminal startup.

Test that Sofa launches by running `$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so`.
Test that soft-robot-control works by running `$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so {$REPO_ROOT}/launch_sofa.py`.

Some more info on these instructions are given [here](https://github.com/SofaDefrost/plugin.SofaPython3/issues/137#issuecomment-571128647) and [here](https://www.sofa-framework.org/community/forum/topic/using-softrobots-sofapython3/).

-----
##### Installation of ROS2

SofaPython3 requires Python 3.8, hence the instructions for ROS2 compatible with python3.8 (which is default for ubuntu 20.04) is provided. 

Download the Debian binaries for [ROS2 Foxy Fitzroy](https://index.ros.org/doc/ros2/Installation/Foxy/) on ubuntu. 
Follow the steps provided in the link to complete the setup of ROS2. Follow steps below for further installation instructions.

ROS2 recommends the usage of colcon for building packages:

```
sudo apt install python3-colcon-common-extensions
pip install -U git+https://github.com/colcon/colcon-common-extensions.git
```

Lark parser is required for ROS2 

```
pip install lark_parser
```

Next setup the ROS2 workspace, follow the instructions of step 1+2 of the [guided tutorial](https://index.ros.org/doc/ros2/Tutorials/Workspace/Creating-A-Workspace/). Optionally you can use `../setup.zsh` instead of `../setup.bash` and you can add it to your shell so that it is sourced upon launching the terminal.

Next we will setup a package (using cmake) to have the service required for using SOFA with cvxpy, for Sequential Convex Programming trajectory optimization problems. Setup is based upon the [Custom ROS2 Interface Page](https://index.ros.org/doc/ros2/Tutorials/Custom-ROS2-Interfaces/):

```
# Navigate to src directory in root of workspace.
ros2 pkg create --build-type ament_cmake soft_robot_control_ros
cd soft_robot_control_ros
mkdir srv
cp {$REPO_DIR}/dependencies/ros/GuSTOsrv.srv srv/
```

Add the following to `CMakeLists.txt` in `${WS_DIR}/src/soft_robot_control_ros`:

```
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(soft_robot_control_ros
  "srv/GuSTOsrv.srv"
 )
```

Add the following to `package.xml`

```
<build_depend>rosidl_default_generators</build_depend>

<exec_depend>rosidl_default_runtime</exec_depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

Build the `soft_robot_control_ros` package. From `${WS_DIR}` run

```
colcon build --packages-select soft_robot_control_ros
```

Within your workspace `${WS_DIR}`, to source it run:

```
. install/setup.bash  # Alternative: . install/setup.zsh
```

To validate that the service is created run the `ros2 interface show` command:

```
ros2 interface show soft_robot_control_ros/srv/GuSTOsrv
```

---

---

## Running soft-robot-control and Sofa

There are two methods to launching simulations using soft-robot-control with Sofa. Both require setting the problem file
in `problem_specification.py`.

1. Run with the simulation GUI. `$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so {$REPO_ROOT}/launch_sofa.py`
2. Run in the background. `python launch_sofa.py` (Requires modifying `sofa_lib_path` to match environment in `launch_sofa.py` file)


---

---

## Setup Environment variables
```
source /home/jjalora/ros2_foxy/install/setup.bash
source /home/jjalora/ros2_ws/install/setup.bash
conda activate sofa
```

## Run the following commands to run minimal example
```
$SOFA_BLD/bin/runSofa -l $SP3_BLD/lib/libSofaPython3.so ~/soft-robot-control/launch_sofa.py

python3 ~/soft-robot-control/examples/diamond/diamond_rompc.py run_rompc_solver
```

## References

[1] F. Faure, C. Duriez, H. Delingette, J. Allard, B. Gilles, S. Marchesseau,
H. Talbot, H. Courtecuisse, G. Bousquet, I. Peterlik, and S. Cotin,
“SOFA: A multi-model framework for interactive physical simulation,”
in Soft Tissue Biomechanical Modeling for Computer Assisted Surgery, 2012. 

[2] E. Coevoet, T. Morales-Bieze, F. Largilliere, Z. Zhang, M. Thieffry,
M. Sanz-Lopez, B. Carrez, D. Marchal, O. Goury, J. Dequidt, and
C. Duriez, “Software toolkit for modeling, simulation, and control of
soft robots,” Advanced Robotics, vol. 31, no. 22, pp. 1208–1224, 2017.
