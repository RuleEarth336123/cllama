# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chunyu123/github/InferX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chunyu123/github/InferX/build

# Include any dependencies generated for this target.
include CMakeFiles/llama.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/llama.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/llama.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/llama.dir/flags.make

CMakeFiles/llama.dir/src/tensor/tensor.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/tensor/tensor.cpp.o: ../src/tensor/tensor.cpp
CMakeFiles/llama.dir/src/tensor/tensor.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/llama.dir/src/tensor/tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/tensor/tensor.cpp.o -MF CMakeFiles/llama.dir/src/tensor/tensor.cpp.o.d -o CMakeFiles/llama.dir/src/tensor/tensor.cpp.o -c /home/chunyu123/github/InferX/src/tensor/tensor.cpp

CMakeFiles/llama.dir/src/tensor/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/tensor/tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/tensor/tensor.cpp > CMakeFiles/llama.dir/src/tensor/tensor.cpp.i

CMakeFiles/llama.dir/src/tensor/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/tensor/tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/tensor/tensor.cpp -o CMakeFiles/llama.dir/src/tensor/tensor.cpp.s

CMakeFiles/llama.dir/src/base/alloc.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/base/alloc.cpp.o: ../src/base/alloc.cpp
CMakeFiles/llama.dir/src/base/alloc.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/llama.dir/src/base/alloc.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/base/alloc.cpp.o -MF CMakeFiles/llama.dir/src/base/alloc.cpp.o.d -o CMakeFiles/llama.dir/src/base/alloc.cpp.o -c /home/chunyu123/github/InferX/src/base/alloc.cpp

CMakeFiles/llama.dir/src/base/alloc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/base/alloc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/base/alloc.cpp > CMakeFiles/llama.dir/src/base/alloc.cpp.i

CMakeFiles/llama.dir/src/base/alloc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/base/alloc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/base/alloc.cpp -o CMakeFiles/llama.dir/src/base/alloc.cpp.s

CMakeFiles/llama.dir/src/base/base.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/base/base.cpp.o: ../src/base/base.cpp
CMakeFiles/llama.dir/src/base/base.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/llama.dir/src/base/base.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/base/base.cpp.o -MF CMakeFiles/llama.dir/src/base/base.cpp.o.d -o CMakeFiles/llama.dir/src/base/base.cpp.o -c /home/chunyu123/github/InferX/src/base/base.cpp

CMakeFiles/llama.dir/src/base/base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/base/base.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/base/base.cpp > CMakeFiles/llama.dir/src/base/base.cpp.i

CMakeFiles/llama.dir/src/base/base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/base/base.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/base/base.cpp -o CMakeFiles/llama.dir/src/base/base.cpp.s

CMakeFiles/llama.dir/src/base/buffer.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/base/buffer.cpp.o: ../src/base/buffer.cpp
CMakeFiles/llama.dir/src/base/buffer.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/llama.dir/src/base/buffer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/base/buffer.cpp.o -MF CMakeFiles/llama.dir/src/base/buffer.cpp.o.d -o CMakeFiles/llama.dir/src/base/buffer.cpp.o -c /home/chunyu123/github/InferX/src/base/buffer.cpp

CMakeFiles/llama.dir/src/base/buffer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/base/buffer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/base/buffer.cpp > CMakeFiles/llama.dir/src/base/buffer.cpp.i

CMakeFiles/llama.dir/src/base/buffer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/base/buffer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/base/buffer.cpp -o CMakeFiles/llama.dir/src/base/buffer.cpp.s

CMakeFiles/llama.dir/src/op/add.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/op/add.cpp.o: ../src/op/add.cpp
CMakeFiles/llama.dir/src/op/add.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/llama.dir/src/op/add.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/op/add.cpp.o -MF CMakeFiles/llama.dir/src/op/add.cpp.o.d -o CMakeFiles/llama.dir/src/op/add.cpp.o -c /home/chunyu123/github/InferX/src/op/add.cpp

CMakeFiles/llama.dir/src/op/add.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/op/add.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/op/add.cpp > CMakeFiles/llama.dir/src/op/add.cpp.i

CMakeFiles/llama.dir/src/op/add.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/op/add.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/op/add.cpp -o CMakeFiles/llama.dir/src/op/add.cpp.s

CMakeFiles/llama.dir/src/op/layer.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/src/op/layer.cpp.o: ../src/op/layer.cpp
CMakeFiles/llama.dir/src/op/layer.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/llama.dir/src/op/layer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/src/op/layer.cpp.o -MF CMakeFiles/llama.dir/src/op/layer.cpp.o.d -o CMakeFiles/llama.dir/src/op/layer.cpp.o -c /home/chunyu123/github/InferX/src/op/layer.cpp

CMakeFiles/llama.dir/src/op/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/src/op/layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/src/op/layer.cpp > CMakeFiles/llama.dir/src/op/layer.cpp.i

CMakeFiles/llama.dir/src/op/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/src/op/layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/src/op/layer.cpp -o CMakeFiles/llama.dir/src/op/layer.cpp.s

CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o: ../kernels/cpu/add_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/add_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/add_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/add_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o: ../kernels/cpu/emb_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/emb_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/emb_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/emb_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o: ../kernels/cpu/matmul_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/matmul_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/matmul_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/matmul_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o: ../kernels/cpu/mha_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/mha_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/mha_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/mha_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o: ../kernels/cpu/rmsnorm_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/rmsnorm_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/rmsnorm_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/rmsnorm_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o: ../kernels/cpu/rope_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/rope_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/rope_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/rope_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o: ../kernels/cpu/scale_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/scale_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/scale_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/scale_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o: ../kernels/cpu/softmax_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/softmax_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/softmax_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/softmax_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.s

CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o: ../kernels/cpu/swiglu_kernel.cpp
CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o -MF CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o.d -o CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o -c /home/chunyu123/github/InferX/kernels/cpu/swiglu_kernel.cpp

CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/cpu/swiglu_kernel.cpp > CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.i

CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/cpu/swiglu_kernel.cpp -o CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.s

CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o: CMakeFiles/llama.dir/flags.make
CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o: ../kernels/kernel_interface.cpp
CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o: CMakeFiles/llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o -MF CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o.d -o CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o -c /home/chunyu123/github/InferX/kernels/kernel_interface.cpp

CMakeFiles/llama.dir/kernels/kernel_interface.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/llama.dir/kernels/kernel_interface.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chunyu123/github/InferX/kernels/kernel_interface.cpp > CMakeFiles/llama.dir/kernels/kernel_interface.cpp.i

CMakeFiles/llama.dir/kernels/kernel_interface.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/llama.dir/kernels/kernel_interface.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chunyu123/github/InferX/kernels/kernel_interface.cpp -o CMakeFiles/llama.dir/kernels/kernel_interface.cpp.s

# Object files for target llama
llama_OBJECTS = \
"CMakeFiles/llama.dir/src/tensor/tensor.cpp.o" \
"CMakeFiles/llama.dir/src/base/alloc.cpp.o" \
"CMakeFiles/llama.dir/src/base/base.cpp.o" \
"CMakeFiles/llama.dir/src/base/buffer.cpp.o" \
"CMakeFiles/llama.dir/src/op/add.cpp.o" \
"CMakeFiles/llama.dir/src/op/layer.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o" \
"CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o"

# External object files for target llama
llama_EXTERNAL_OBJECTS =

../lib/libllama.so: CMakeFiles/llama.dir/src/tensor/tensor.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/src/base/alloc.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/src/base/base.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/src/base/buffer.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/src/op/add.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/src/op/layer.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/add_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/emb_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/matmul_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/mha_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/rmsnorm_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/rope_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/scale_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/softmax_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/cpu/swiglu_kernel.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/kernels/kernel_interface.cpp.o
../lib/libllama.so: CMakeFiles/llama.dir/build.make
../lib/libllama.so: /usr/local/lib/libglog.so.0.8.0
../lib/libllama.so: /usr/lib/x86_64-linux-gnu/libarmadillo.so
../lib/libllama.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
../lib/libllama.so: CMakeFiles/llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chunyu123/github/InferX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Linking CXX shared library ../lib/libllama.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/llama.dir/build: ../lib/libllama.so
.PHONY : CMakeFiles/llama.dir/build

CMakeFiles/llama.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/llama.dir/cmake_clean.cmake
.PHONY : CMakeFiles/llama.dir/clean

CMakeFiles/llama.dir/depend:
	cd /home/chunyu123/github/InferX/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chunyu123/github/InferX /home/chunyu123/github/InferX /home/chunyu123/github/InferX/build /home/chunyu123/github/InferX/build /home/chunyu123/github/InferX/build/CMakeFiles/llama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/llama.dir/depend

