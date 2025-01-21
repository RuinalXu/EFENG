# EFENG

---

This is official repository for "*[Explainable Fake News Detection with Generated Rationales from Large Language Models](论文链接)*".

## Datasets

The origin news texts data used in the experiments can be found at the official links provided:
- The LIAR dataset can be accessed at the publisher's site, get the origin [liar dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) project and cite the [paper](https://www.aclweb.org/anthology/P17-2067.pdf).
- The GossipCop dataset can be found by following the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) project and cite the [paper](https://arxiv.org/pdf/1809.01286).

**Note**: The data used in use is not the origin Liar and GossipCop dataset, but the data obtained through step [Explainable Features](#explainable-features).

## Explainable Features

The explainable features are generated using the Qwen model or a locally deployed LLaMa3 model. 
You need to set your API_KEY in the [rationale_generator.py](./explainable_features/rationale_generator.py) file and modify the LLM instructions according to the provided template.

## How to run

### Requirements

- python==3.10.14
- CUDA==13.4
- All other dependencies can be obtained in [requirements.txt](./requirements.txt).
  - installed by running the following command: `pip install -r requirements.txt`

### PLMs
The [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model is used.
You can download the pre-trained bert-base-uncased model locally and place it in the `model` directory.
Then, modify the command-line argument `bert_path` to point to the correct location.

### Run shell
You can run this model through `run.sh` for datasets.


**Note**: If you are using relative paths, make sure to set the correct working directory.