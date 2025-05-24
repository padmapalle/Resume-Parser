import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from spacy.scorer import Scorer

# Load your trained spaCy model
nlp = spacy.load(r"C:\Users\padma\OneDrive\Desktop\resume parser\output\model-best")

# Load the test data
doc_bin = DocBin().from_disk(r"D:\jyothika\mini project 2\resume parser\test_data.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Initialize the scorer
scorer = Scorer()

# Create examples from the docs
examples = []
for doc in docs:
    example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
    examples.append(example)

# Score the model
scores = nlp.evaluate(examples)

# Print the results
print("Precision:", scores["ents_p"])
print("Recall:", scores["ents_r"])
print("F1-score:", scores["ents_f"])