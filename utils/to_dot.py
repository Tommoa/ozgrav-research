#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess
import io
from itertools import chain, repeat
from typing import List


def get_dot(input_file: io.IOBase) -> io.StringIO:
    """
    Generate a `DOT` callgraph from an LLVM IR file.

    :param io.IOBase input_file: The input LLVM IR file
    :returns io.TextIO: The resulting `DOT` graph
    """
    opt = subprocess.Popen([
        "opt",
        "-analyze",
        "-std-link-opts",
        "-dot-callgraph"],
        stdin=input_file,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    opt.wait()
    dotfile = pathlib.Path("callgraph.dot")
    ret = None
    with dotfile.open("r") as f:
        filt = subprocess.Popen([
            "c++filt",
            "-n",
            "-p"],
            stdin=f,
            stdout=subprocess.PIPE,
            text=True)
        ret = io.StringIO(filt.communicate()[0])
    dotfile.unlink()
    return ret


def parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "A helper to turn LLVM IR to a DOT file"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "input",
        type=argparse.FileType(
            'r'),
        nargs="*",
        default=None,
        action="extend")
    group.add_argument(
        "--directory",
        "-d",
        type=pathlib.Path,
        default=pathlib.Path("."),
        nargs="?")
    return parser


def run(args: argparse.Namespace) -> List[pathlib.Path]:
    import sys
    ret = []
    inputs = args.input or list(map(open, args.directory.glob("*.ll")))
    for i in inputs:
        try:
            ret.append(get_dot(i))
        except Exception as e:
            print("Doing `{}` failed with exception: {}".format(
                i.name, e), file=sys.stderr)
    return ret


def main(args: argparse.Namespace):
    import sys
    all_files = run(args)
    if len(all_files) == 0:
        print("No files processed", file=sys.stderr)
    for output in all_files:
        print(output.read())
    sys.stdout.flush()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parsed = parser(argparser).parse_args()
    main(parsed)
