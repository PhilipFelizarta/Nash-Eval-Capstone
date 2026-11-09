"""Builds the nash_mcts extension into cpp/mcts/ (CMake + pybind11, at most 2 parallel jobs).

	python cpp/mcts/build.py
"""
import os
import subprocess
import sys

import pybind11

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")


def main():
	jobs = os.environ.get("NASH_MCTS_JOBS", "2")
	subprocess.check_call([
		"cmake", "-S", HERE, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
		f"-Dpybind11_DIR={pybind11.get_cmake_dir()}", f"-DPython_EXECUTABLE={sys.executable}",
	])
	subprocess.check_call(["cmake", "--build", BUILD, "-j", jobs])


if __name__ == "__main__":
	main()
