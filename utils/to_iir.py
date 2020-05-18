#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
from itertools import chain, repeat
from typing import List, Tuple


def make_process(
        input_file: pathlib.Path,
        includes: List[pathlib.Path] = [],
        cuda_path: pathlib.Path = os.getenv("CUDA_PATH") or "",
        cuda_arch: str = "sm_61",
        flags: List[str] = [],
) -> Tuple[pathlib.PurePath, subprocess.Popen]:
    """
    Run `clang` on an input file, and return the path to the resulting
    generated iir file and the process.

    This function calls `clang` instead of `clang++` because there are a number
    of type casts in SPIIR that are fine using C, but not with C++.

    :param pathlib.Path input_file: The file to compile
    :param List[pathlib.Path] includes: A list of additional directories to
    include
    :param pathlib.Path cuda_path: The path to the CUDA install directory.
    :param str cuda_arch: The architecture of the resulting CUDA device.
    :param List[str] flags: Any additional flags
    :returns Tuple[pathlib.PurePath, subprocess.Popen]: The resulting IIR file
    and `clang` process
    """
    includes = [
        ".",
        "/fred/oz016/gwdc_spiir_install/dependencies/include/",
        "/fred/oz016/gwdc_spiir_install/dependencies/include/glib-2.0/",
        "/fred/oz016/gwdc_spiir_install/dependencies/include/gstreamer-0.10/",
        "/fred/oz016/gwdc_spiir_install/dependencies/lib64/glib-2.0/include/",
        "/fred/oz016/gwdc_spiir_pipeline_codebase/scripts_n_things/build/master/install/include/",
        "/usr/include/libxml2/",
    ] + includes
    cuda_path = cuda_path or pathlib.Path("/usr/local/cuda-10.0.130/")
    flags = ["-S", "-emit-llvm"] + flags
    output: pathlib.PurePath = pathlib.PurePath(input_file.stem+".ll")
    clang = subprocess.Popen(
        ["clang"]
        + list(chain.from_iterable(zip(repeat("-I", len(includes)), includes)))
        + flags
        + ["--cuda-path={}".format(cuda_path), "--cuda-gpu-arch={}".format(cuda_arch)]
        + [input_file])
    return (output, clang)


def parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "A helper to turn C or CUDA to LLVM IIR"
    parser.add_argument(
        "--include",
        "-I",
        type=pathlib.Path,
        nargs="*",
        default=[],
        action="append")
    parser.add_argument(
        "--cuda-path",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("CUDA_PATH") or ""),
        nargs="?")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        nargs="+",
        action="append")
    group.add_argument(
        "--directory",
        "-d",
        type=pathlib.Path,
        default=pathlib.Path("."),
        nargs="?")
    parser.add_argument(
        "--cuda-device",
        type=str,
        nargs="?",
        default="sm_61")
    parser.add_argument(
        "--flags",
        "-f",
        nargs="*",
        default=[],
        action="append")
    return parser


def run(args: argparse.Namespace) -> List[pathlib.Path]:
    import sys
    import glob
    ret = []
    inputs = list(chain.from_iterable(args.input or glob.glob(str(args.directory)+"/*.c{,pp,u}")))
    for i in inputs:
        try:
            path, process=make_process(
                i,
                args.include,
                args.cuda_path,
                args.cuda_device,
                args.flags)
            process.wait()
            ret.append(path)
        except Exception as e:
            print("Doing `{}` failed with exception: {}".format(
                i, e), file=sys.stderr)
    return ret


def main(args: argparse.Namespace):
    import sys
    all_files=run(args)
    if len(all_files) == 0:
        print("No files processed", file=sys.stderr)
    for output in all_files:
        print(output, flush=False)
    sys.stdout.flush()


if __name__ == "__main__":
    argparser=argparse.ArgumentParser()
    parsed=parser(argparser).parse_args()
    main(parsed)
