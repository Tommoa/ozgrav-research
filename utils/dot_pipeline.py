import argparse
import io
import pathlib
from typing import List

from .combine_dot import combine_dot
from .to_dot import get_dot


def pipeline(input_files: List[io.IOBase], output: io.IOBase):
    combine_dot(map(get_dot, input_files), output)


def parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    import sys
    parser.description = "A helper to turn LLVM IR to a DOT file"

    parser.add_argument("--output",
                        "-o",
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        nargs=1)

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


def main(args: argparse.Namespace):
    import sys
    pipeline(args.input or list(map(open, args.directory.glob("*.ll"))), args.output)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    parsed = parser(argparser).parse_args()
    main(parsed)
