# Multi-Session Ground Texture SLAM Evaluation #

Code to evaluate the efficacy of various methods for operating in multi-session, low-dynamic-change ground texture SLAM
environment. This also generates all the figures and values added to the associated paper.

## Distribution Statement ##

**Distribution Statement A:** Approved for public release; distribution is unlimited, as submitted under NAVAIR Public
Release Authorization 2025-0098. The views expressed here are those of the authors and do not reflect the official
policy or position of the United States Navy, Department of Defense, or United States Government.

## License ##

See [LICENSE](LICENSE.md) and [INTENT](INTENT.md) for information.

## Citation ##

If using this in your academic work, please cite us. Exact citation is TBD, as this is part of a currently under-review
paper. This will be updated if accepted. Reach out directly if you are using prior to this being updated.

## Usage ##

To reproduce the results, follow these steps:

1. Build the Docker image for our related work:
[https://github.com/Navy-RISE-Lab/ground-texture-slam](https://github.com/Navy-RISE-Lab/ground-texture-slam)
2. Download the data:
[https://huggingface.co/datasets/khartrise/multi-session-ground-texture-data](https://huggingface.co/datasets/khartrise/multi-session-ground-texture-data)
3. Build the Dev Container container for this project. Make sure the definitions in
[devcontainer.json](.devcontainer/devcontainer.json) for the data and base image tag are up to date.
4. Once the container is build, run `dvc repro` to run all the experiments. This will take a long time to run. When
finished, all results will be in the *output* folder.

If not following these steps, you can make your own environment. Just consider these points:

* You will need the Ground Texture SLAM project in that environment, as this one leverages several components from it.
* The [requirements.txt](requirements.txt) has all the needed Python dependencies.
* This uses [DVC](https://dvc.org/doc/start) to define all the stages involved in running the experiments. In general,
there are separate stages for running the experiments and making the plots.
* If using different data, make sure to follow the format shown in the data linked above. It assumes a certain
structure. Also, update the *paths* field in [params.yaml](params.yaml) to point to each session of data.
