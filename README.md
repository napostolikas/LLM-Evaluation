# Evaluation of LLMs using custom data

This thesis project focuses on evaluating three different LLMs (i.e. Starling-LM-Alpha, OpenHermes-Mistral-2.5 and Openchat-3.5) on downstream NLP tasks. The tasks are Natural Language Inference (NLI), Named Entity Recognition (NER) and Open-book Question Answering (QA). For each task we test different moods, voices, prompt templates in various few-shot scenarios (i.e. 0, 1 and 3). 

For NER and QA we also implemented a custom evaluation pipeline based on conversational AI. Each model has its own folder where you can find the script for the inference for every task. In each folder, you can also find the scripts for evaluating NLI and NER (with the traditional method: dict-to-dict comparison as we prompt the model to provide the output in the desired format).

For the conversational AI pipeline the scripts are located in the main directory. Since we use an LLM in the pipeline, human evaluations is essential. Thus there are two scripts that human-perform this task and compare the kappa score between the pipeline and the annotator-performance.
