# OzGrav Research

[![DOI](https://zenodo.org/badge/250756435.svg)](https://zenodo.org/badge/latestdoi/250756435)

This repository is a listing of my time doing research for the OzGrav team at UWA.

OzGrav researches gravitational waves, and attempts to use the data taken from various gravitation wave detectors (primarily LIGO in the US) to determine the direction of various waves so that their sources may be studied. The UWA team has developed a pipeline for analysis called SPIIR (summed parallel infinite impulse response) to provide low latency gravitational wave detection.

My research project primarily focuses on developing methods to make it easier for the pipeline to be modified to work with any number of input detectors.

As a result of this, there are now future plans to filter or ignore some detectors for detection, but still use those detectors for other parts of the pipeline.
To give a proper idea, the below is the data flow for the current pipeline:

![Current pipeline data flow](resources/current_pipeline.png)

The plan is to implement this data flow:

![New pipeline data flow](resources/new_pipeline.png)

## Components

There will be multiple components of this research repository, and links to all of them will be found in this section.

- [Proposal](https://tommoa.github.io/ozgrav-research/proposal.pdf)
- [Progress Report (2020/05/29)](https://tommoa.github.io/ozgrav-research/progress-report.pdf)
- [Callgraph for postprocessing](https://github.com/Tommoa/ozgrav-research/blob/master/resources/callgraph.png)
- [Complexity analysis of postprocessing](https://tommoa.github.io/ozgrav-research/analysis.pdf)
- Seminar
  - [Seminar abstract](https://tommoa.github.io/ozgrav-research/abstract.pdf)
  - [Seminar presentation](https://tommoa.github.io/ozgrav-research/seminar-presentation.pdf)
  - [Seminar talk](https://tommoa.github.io/ozgrav-research/seminar-talk.pdf)
- [Dissertation](https://tommoa.github.io/ozgrav-research/dissertation.pdf)

## Building benchmarks

- You can build the benchmarks with:
`make -C benchmarking`
- If you want to use `nvcc` (NVIDIA's CUDA compiler) instead of `clang`, run:
`make -C benchmarking nvcc`

### Dependencies

- CUDA > 9.0
- clang > 8.0
- Intel TBB (`clang` only)
