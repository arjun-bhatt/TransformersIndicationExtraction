# TransformersIndicationExtraction

Features:
 - new dataset of sentences labelled with their purpose: multiclass labels for miscellaneous (0), indication (1), contra-indication (2), side effects (3), and instructions for use (4)
 - validation of this dataset from established DrugBank sources
 - implementations of popular transformers models evaluated on this dataset

Repository Contents:
- preprocessing_labeling_xml: a notebook that uses the human Rx drug labeling information from the FDA as a corpus, tokenizes sentences from those labels using a custom tokenizer 
- LSTM_Model: a notebook that trains & evaluates a bi-directional LSTM model on top of that dataset
- Transformers_Indication_Extraction: a notebook that trains and evalutes two transformers models individually on that dataset, then stacks the hidden state representations of each sentence from each model, and trains and evaluates a new model based off those concatenated representations. Also validates dataset using sentences extracted from Drugbank.
