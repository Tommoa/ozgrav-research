#!/usr/bin/env python3
import argparse
import io
from typing import Iterator, Union

import matplotlib.pyplot as plt
import networkx as nx

# Unfortunately `mypy` doesn't work on this file because `networkx` and
# `matplotlib` do not have `.pyi` files.


def combine_dot(dots: Iterator[Union[io.IOBase, str]],
                output: Union[io.IOBase, str]) -> nx.DiGraph:
    """
    A helper function to combine a number of GraphViz DOT files together to
    create a single DOT file.

    :param Iterator[Union[io.IOBase, str]] dots: An iterator on the DOT files
    to combine. They can either be paths, or opened io.IOBase instances.
    :param Union[io.IOBase, str] output: Where to write the resulting DOT file
    to. Can be a path or an io.IOBase instance.
    :return nx.DiGraph: The resulting directed graph.
    """
    dots = map(nx.nx_pydot.read_dot, dots)
    dots = list(
        map(
            lambda graph: nx.DiGraph(  # Make there be at most 1 edge between nodes
                # Change the node names from "Node0x..." to the function names
                nx.relabel_nodes(graph, nx.get_node_attributes(graph, 'label'))
            ), dots))

    # Remove all the nodes which are either still "Node0x..." or are "external node"
    for graph in dots:
        graph.remove_nodes_from(
            list(
                filter(lambda name: 'Node' in name or 'node' in name,
                       graph.nodes())))

    # Combine the graphs together
    composed = nx.compose_all(dots)
    # Remove all nodes that don't have a connecting edge
    # examples of nodes like this would be dead code, builtin functions or
    # functions that get inlined
    composed.remove_nodes_from(
        node for node, degree in dict(composed.degree()).items() if degree < 1)

    # Write the resulting graph to an output
    nx.nx_pydot.write_dot(composed, output)
    return composed


def _parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    import sys
    parser.add_argument("--output",
                        "-o",
                        type=argparse.FileType('w'),
                        default=sys.stdout,
                        nargs=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--directory", "-d", type=str, default=".", nargs='?')
    group.add_argument("inputs", type=argparse.FileType(
        'r'), nargs="*", default=None)

    parser.add_argument("--display", action="store_true")
    parser.description = "Combine a number of GraphViz DOT files to a single file"
    return parser


def _main(args: argparse.Namespace):
    import glob
    dots = args.inputs
    if not dots:
        dots = glob.glob(args.directory + '/*.dot')
    composed = combine_dot(dots, args.output)
    if args.display:
        nx.draw(composed, with_labels=True)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsed = _parser(parser).parse_args()
    _main(parsed)
