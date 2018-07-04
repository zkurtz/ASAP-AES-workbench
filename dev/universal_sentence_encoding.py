import pdb
import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

from asap_essay_scoring import data
from asap_essay_scoring import utils

# Sentize first essay for first prompt
df = data.read_raw_csv(utils.data_path('training_set_rel3.tsv')).iloc[0]
essay = df.essay[0]

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(essay)


# Sentize first prompt
prompt = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
pdb.set_trace()


embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

#session.run(embeddings)