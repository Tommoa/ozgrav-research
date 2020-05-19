#!/usr/bin/env python3
import argparse
import subprocess
from .combine_dot import parser as make_combine_parser, main as combine_main
from .to_iir import parser as make_iir_parser, main as iir_main
from .to_dot import parser as make_dot_parser, main as dot_main
from .dot_pipeline import parser as make_pipeline_parser, main as pipeline_main

parser = argparse.ArgumentParser(
        "utils",
        description="Utilities to help with Tom Almeida's GENG5551 research.")
parser.set_defaults(func=parser.print_help)

subparsers = parser.add_subparsers(
        title="subcommands",
        help="subcommand help")

combine_parser = subparsers.add_parser("combine", aliases=["co"])
make_combine_parser(combine_parser)
combine_parser.set_defaults(func=combine_main)

iir_parser = subparsers.add_parser("iir")
make_iir_parser(iir_parser)
iir_parser.set_defaults(func=iir_main)

dot_parser = subparsers.add_parser("dot")
make_dot_parser(dot_parser)
dot_parser.set_defaults(func=dot_main)

pipeline_parser = subparsers.add_parser("pipeline")
make_pipeline_parser(pipeline_parser)
pipeline_parser.set_defaults(func=pipeline_main)

args = parser.parse_args()
if args.func == parser.print_help:
    args.func()
else:
    args.func(args)
