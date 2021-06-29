## TransformersIndicationExtraction
The code behind a manuscript presenting (and validating via DrugBank) a new dataset based off of FDA human Rx drug labeling documents, evaluating popular models for sequence classification from transformers huggingface library, and training and evaluating a separate Bi-directional LSTM NN on that dataset.

## Motivation
Extracting computable indications, i.e. drug-disease treatment relationships, from narrative drug resources is the key for building a gold standard drug indication repository. The two steps to the extraction problem are disease named-entity recognition (NER) to identify disease mentions from a free-text description and disease classification to distinguish indications from other disease mentions in the description. While there exist many tools for disease NER, disease classification is mostly achieved through human annotations. The development of artificial intelligence to identify sentences that constitute an indication for a given drug would allow for a significant reduction in human annotation costs, and could aid in future attempts for drug repositioning.
 
## Demo

[Not terribly flashy I'm afraid](https://imgur.com/a/dTrYPOj)

## Tech/framework used
Transformers HuggingFace library for BERT, RoBERTa, Albert, Distilbert, & Biobert models
Keras & Tensorflow for LSTM Model
NLTK for development of custom tokenizer

## Features
- new dataset of sentences labelled with their purpose: multiclass labels for miscellaneous (0), indication (1), contra-indication (2), side effects (3), and instructions for use (4)
- validation of this dataset from established DrugBank sources
- implementations of popular transformers models evaluated on this dataset
- presents an easy-to-use framework for ensemble learning based off transformers models

## Contents:
- preprocessing_labeling_xml: a notebook that uses the human Rx drug labeling information from the FDA as a corpus, tokenizes sentences from those labels using a custom tokenizer 
- LSTM_Model: a notebook that trains & evaluates a bi-directional LSTM model on top of that dataset
- Transformers_Indication_Extraction: a notebook that trains and evalutes two transformers models individually on that dataset, then stacks the hidden state representations of each sentence from each model, and trains and evaluates a new model based off those concatenated representations. Also validates dataset using sentences extracted from Drugbank.
- **AnnotatedDataset** : Composed of 7231 labeled sentences extracted from the Indications & Usage section of FDA Rx Labeling Documents. During the extraction process, sentences were lemmatized, stop words removed, and punctuation/digit/two character words removed as well. Sentences are labeled numerically 1-5, where:
- - 0: clinical observation (1673)
- - 1: indication (4297)
- - 2: contraindication (492)
- - 3: side effect (68)
- - 4: usage instruction (701)
- - These assignments were verified by 3 pharmacologists. Sentences were collapsed into indication (1) or non-indication (0, 2-4) for the purposes of classification. An 80/20 train/test split was used.
- drugbank-indication: Used to validate models trained on AnnotatedDataset; this is composed of solely indications extracted from Drugbank. Useful for ensuring models are picking up on patterns present in sentences that represent indications in the general case rather than sentences that represent indications in FDA documents. 



## Code Example

Using transformers to create models

```
from transformers import BertTokenizer, AutoTokenizer, AlbertTokenizer, DistilBertTokenizer
from transformers import BertForSequenceClassification, AutoModelWithLMHead, RobertaConfig, BertForMaskedLM, BertConfig, AlbertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
tokenizer2 = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
model2 = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True)

epochs = 1
epochs2 = 1
```

Easily extracting final hidden state representations for each sentence

```
train_sentences, train_labels, test_sentences, test_labels, device = environment_setup()
train_dataloader = tokenize_and_organize_data(tokenizer, train_sentences, train_labels)
validation_dataloader = tokenize_and_organize_data(tokenizer, test_sentences, test_labels)
scheduler, optimizer = load_model(model, epochs, train_dataloader)
_, flat_master_truth, trained_model, hidden_states = train_model(model, epochs, train_dataloader, validation_dataloader, scheduler, optimizer, device)
```

for both models

```
train_dataloader2 = tokenize_and_organize_data(tokenizer, train_sentences, train_labels)
validation_dataloader2 = tokenize_and_organize_data(tokenizer, test_sentences, test_labels)
scheduler2, optimizer2 = load_model(model2, epochs2, train_dataloader2)
_, flat_master_truth2, trained_model2, hidden_states2 = train_model(model2, epochs2, train_dataloader2, train_dataloader2, scheduler2, optimizer2, device)
```

Then stacking those hidden states & training a new model to take the concatenated hidden states and make predictions with the advantages of features from both models.

```
joined_lists = np.column_stack([flat_hidden_states, flat_hidden_states2])
```


## Installation
Port the .ipynb files over to Google Colab, change the PATHs to point to locations in your drive, [mount your drive in colab](https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/), then you're done!


## Credits
Many thanks to Dr. Zhichao Liu, Dr. Xiangwen Liu, and Dr. Weida Tong of the NCTR.
And with gratitude towards the huggingface transformers team.

## License

MIT © [Arjun Bhatt]()
