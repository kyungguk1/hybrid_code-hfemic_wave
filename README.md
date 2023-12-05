# Archive of the hybrid code used for HFEMIC wave simulations

The input parameters for the cases examined in the paper are under `00-runs`.
Visit one of the subdirectories and follow the build instruction below.

## Build Instruction

The project uses CMake as a build environment. Building the targets requires **MPI** and **hdf5** libraries available.

> Avoid compiling with OpenMPI. This implementation seems to misbehave when communicating struct-like data.

Once all the dependencies are met, follow the steps below to build executables:

1. Clone the project

Follow the instruction in the project page.

2. Make a build directory

```shell
mkdir build && cd build
```

3. Generate the build configurations

```shell
cmake -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_IPO=On \
    -DPIC_INPUT_DIR=${PATH_TO_PIC_SIMULATION_INPUT_HEADER} \
    -G "Ninja" ${PROJECT_PATH}
```

- If `ninja` is not available, replace `"Ninja"` with `"Unix Makefiles"`.
- `PROJECT_PATH` refers to the project directory you just cloned.
- Set `PATH_TO_PIC_SIMULATION_INPUT_HEADER` to the path to a directory containing `Input.h`.

4. Build the executables

```shell
ninja
```

The executable built is available at `src/hybrid_1d/hybrid_1d`.

## LICENSE

All my contributions are provided under the BSD-2 Clause license (see [LICENSE](LICENSE)),
*except* the **Faddeeva** functions (found in
[`Faddeeva.hh`](src/LibPIC/PIC/Misc/Faddeeva.hh) and
[`Faddeeva.cc`](src/LibPIC/PIC/Misc/Faddeeva.cc))
which are provided under the MIT license and
available at [http://ab-initio.mit.edu/Faddeeva](http://ab-initio.mit.edu/Faddeeva).
