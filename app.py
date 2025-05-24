from flask import Flask, render_template, request
import spacy
import pickle
import pdfplumber
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub('[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def classification(txt):
    clf = pickle.load(open('clf.pkl', 'rb'))

    # Clean the input resume
    cleanedResume = cleanResume(txt)

    # Transform the cleaned resume using the trained TF-IDF vectorizer
    inputFeatures = tfidf.transform([cleanedResume])

    # Make the prediction using the loaded classifier
    predictionId = clf.predict(inputFeatures)[0]

    # Map category ID to category name
    categoryMapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }
    categoryName = categoryMapping.get(predictionId, "Unknown")

    print("Predicted Category:", categoryName)
    return categoryName

app = Flask(__name__)

# Load SpaCy model for label extraction
nlp = spacy.load(r"C:\Users\padma\OneDrive\Desktop\resume parser\output\model-best")

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    if request.method == "POST":
        # Check if file is provided
        if "resume" not in request.files:
            return render_template("upload.html", error="No file provided")
        
        resume_file = request.files["resume"]
        
        # Check if file is PDF
        if resume_file.filename.endswith(".pdf"):
            # Read text from PDF
            with pdfplumber.open(resume_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            
            # Process text using SpaCy model for label extraction
            text = text.strip()
            text = ' '.join(text.split())
            text = text.replace('➢', '')

            doc = nlp(text)
            labels = {ent.label_: ent.text for ent in doc.ents}
            
            # Process text using domain classification model
            domain = classification(text)
            
            return render_template("result.html", labels=labels, domain=domain)
        else:
            return render_template("upload.html", error="File must be a PDF")
    
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
