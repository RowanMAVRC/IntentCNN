import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4" 
import torch


"""Start by loading the IMDb dataset from the 🤗 Datasets library"""
from datasets import load_dataset
imdb = load_dataset("imdb")
"""Then take a look at an example:"""
print(imdb["test"][0])

"""The next step is to load a MobileBERT tokenizer to preprocess the `text` field"""
from transformers import AutoTokenizer, MobileBertTokenizer
# tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
# tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased", max_length=512)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

"""Create a preprocessing function to tokenize `text` and truncate sequences to be no longer than MobileBERT's maximum input length"""
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
    # return tokenizer(examples["text"], truncation=True)

"""To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up `map` by setting `batched=True` to process multiple elements of the dataset at once"""
tokenized_imdb = imdb.map(preprocess_function, batched=True)

"""Now create a batch of examples using [DataCollatorWithPadding](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithPadding). It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length."""
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric)"""
import evaluate
accuracy = evaluate.load("accuracy")

"""Then create a function that passes your predictions and labels to [compute](https://huggingface.co/docs/evaluate/main/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the accuracy:"""
import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
"""Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training."""

"""Before you start training your model, create a map of the expected ids to their labels with `id2label` and `label2id`"""
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

"""You're ready to start training your model now! Load MobileBERT with [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification) along with the number of expected labels, and the label mappings"""
from transformers import TrainingArguments, Trainer, MobileBertModel, MobileBertForSequenceClassification
# model = MobileBertModel.from_pretrained(
#                                             "google/mobilebert-uncased",num_labels=2,id2label=id2label,label2id=label2id
#                                         )
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased",num_labels=2,id2label=id2label,label2id=label2id)
# model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased")

training_args = TrainingArguments(
    output_dir="mobileBERT_trial1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.02,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_token='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'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

"""Once training is completed, share your model to the Hub with the [push_to_hub()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model"""
trainer.push_to_hub()

"""
Great, now that you've finetuned a model, you can use it for inference!
Grab some text you'd like to run inference on
"""
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

"""The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for sentiment analysis with your model, and pass your text to it"""
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="cippppy/mobileBERT_trial1")
classifier(text)

"""
You can also manually replicate the results of the `pipeline` if you'd like
Tokenize the text and return PyTorch tensors
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("cippppy/mobileBERT_trial1")
inputs = tokenizer(text, return_tensors="pt")

"""Pass your inputs to the model and return the `logits`"""
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
with torch.no_grad():
    logits = model(**inputs).logits

"""Get the class with the highest probability, and use the model's `id2label` mapping to convert it to a text label"""
predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])