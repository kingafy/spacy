# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:53:19 2019

@author: Anshuman_Mahapatra
"""

"Can you please update me status of my Invoice # 123456789."
"Please provide me update on the Invoice status. Invoice # is 123456789"
"Provide status on invoice 1234567890 and attachments."
"I need to know the status of my invoice #5467879"
"I need to know the status of my invoice #AB1234"
"Please send me the update on invoice CD123456"

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:52:40 2019

@author: Anshuman_Mahapatra
"""

#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABEL = 'INVOICE'

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    ("Can you please update me status of my Invoice # 123456789.", {
        'entities': [(48, 58, 'INVOICE')]
    }),

    ("Please provide me update on the Invoice status. Invoice # is 123456789", {
        'entities': [(61, 70, 'INVOICE')]
    }),

    ("Provide status on invoice 1234567890 and attachments.", {
        'entities': [(26, 37, 'INVOICE')]
    }),

    ("I need to know the status of my invoice #5467879", {
        'entities': [(41, 49, 'INVOICE')]
    }),

    ("I need to know the status of my invoice #AB1234", {
        'entities': [(41, 47, 'INVOICE')]
    }),

    ("Please send me the update on invoice CD123456", {
        'entities': [(37,45, 'INVOICE')]
    })
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='en_core_web_sm', new_model_name='invoice', output_dir='D:/Data Science/POC/spacy/INVOICE', n_iter=100):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
        print("Output of pretrained model")
        doc = nlp("Can you please update me status of my Invoice # 123456789.")
        for ent in doc.ents:
            print(ent.label_,ent.text)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')
    ##before adding new label
    print(nlp.entity.move_names)
    ner.add_label(LABEL)   # add new entity label to entity recognizer
    ##check labels existing after addition
    print(nlp.entity.move_names)
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # test the trained model
    print(nlp)
    test_text = 'Can you please update me status of my Invoice # 123456789.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print("Post Training")
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print("After Training from Saved model")
            print(ent.label_, ent.text)
            

if __name__ == '__main__':
    plac.call(main)
    
    
nlp3 = spacy.load('D:/Data Science/POC/spacy/INVOICE')
doc2 = nlp3("Please let me know the status of Invoice AB12345.")
for ent in doc2.ents:
            print(ent.label_, ent.text)
            
 
nlp3 = spacy.load('D:/Data Science/POC/spacy/INVOICE')
doc2 = nlp3("Barrack Obama wants $100000 to be credited to his account from  Pakistan.")
for ent in doc2.ents:
            print(ent.label_, ent.text) 
            
'''
from spacy import displacy
displacy.serve(doc2, style='ent')
##nlp3.entity.move_names

'''