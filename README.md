!pip install -qU PyPDF2 python-docx pypdf transformers sentence-transformers scikit-learn
!pip install -qU --upgrade langchain-core langchain-community
     
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.0/42.0 kB 1.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.3/11.3 MB 28.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.5/9.5 MB 57.8 MB/s eta 0:00:00

import os
import PyPDF2
import docx
import json
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
from google.colab import files
from IPython.display import display, HTML
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
     

class JobDescription:
    def __init__(self, title: str, description: str, requirements: List[str], preferred_skills: List[str] = None):
        self.title = title
        self.description = description
        self.requirements = requirements
        self.preferred_skills = preferred_skills or []

    def to_dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "requirements": self.requirements,
            "preferred_skills": self.preferred_skills
        }

class Candidate:
    def __init__(self, id: str, name: str, email: str, score: float, summary: str,
                 skills: List[str], experience: str, resume_text: str):
        self.id = id
        self.name = name
        self.email = email
        self.score = score
        self.summary = summary
        self.skills = skills
        self.experience = experience
        self.resume_text = resume_text

    def display(self):
        display(HTML(f"""
        
            👤 {self.name} - Score: {self.score}/100
            📧 Email: {self.email}
            📋 Summary: {self.summary}
            🛠️ Skills: {', '.join(self.skills[:10])}{'...' if len(self.skills) > 10 else ''}
            💼 Experience Highlight: {self.experience[:200]}...
        
        """))
     

class FreeNLPProcessor:
    def __init__(self):
        print("🔄 Loading free NLP models...")
        try:
            # Lightweight model for summarization
            self.summarizer = pipeline("summarization",
                                     model="sshleifer/distilbart-cnn-12-6",
                                     tokenizer="sshleifer/distilbart-cnn-12-6",
                                     framework="pt")
        except:
            self.summarizer = None
            print("⚠️ Summarization model not available, using fallback methods")

        # Sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.sentence_model = None
            print("⚠️ Sentence transformer not available, using TF-IDF")

        print("✅ NLP models initialized")
     

class ResumeProcessor:
    def __init__(self):
        self.nlp_processor = FreeNLPProcessor()
        self.skills_keywords = self._load_skills_keywords()

    def _load_skills_keywords(self):
        """Common technical skills for matching"""
        return {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'data': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit', 'matplotlib', 'seaborn'],
            'devops': ['git', 'ci/cd', 'ansible', 'chef', 'puppet', 'linux', 'bash']
        }

    def extract_text(self, file_content, filename):
        """Extract text from uploaded file"""
        try:
            if filename.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            elif filename.lower().endswith(('.docx', '.doc')):
                doc = docx.Document(io.BytesIO(file_content))
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                return file_content.decode('utf-8')
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def extract_info_regex(self, text):
        """Extract basic info using regex patterns"""
        # Extract name (simple pattern)
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text[:500])
        name = name_match.group(1) if name_match else "Unknown Candidate"

        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        email = email_match.group() if email_match else "no-email@example.com"

        return name, email

    def extract_skills(self, text):
        """Extract skills using keyword matching"""
        text_lower = text.lower()
        found_skills = []

        for category, skills in self.skills_keywords.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)

        return list(set(found_skills))  # Remove duplicates

    def calculate_score(self, resume_text, job_description):
        """Calculate match score using TF-IDF or semantic similarity"""
        # Combine job requirements
        job_text = f"{job_description.title} {job_description.description} {' '.join(job_description.requirements)}"

        if self.nlp_processor.sentence_model:
            # Use sentence transformers for better semantic matching
            resume_embedding = self.nlp_processor.sentence_model.encode(resume_text, convert_to_tensor=True)
            job_embedding = self.nlp_processor.sentence_model.encode(job_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
            score = min(100, max(0, similarity * 100))
        else:
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer().fit_transform([resume_text, job_text])
            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
            score = min(100, max(0, similarity * 100))

        return round(score, 1)

    def generate_summary(self, resume_text, job_description):
        """Generate candidate summary"""
        try:
            if self.nlp_processor.summarizer:
                # Use the summarization model
                summary = self.nlp_processor.summarizer(
                    resume_text[:1024],  # Limit input length
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                return summary
        except:
            pass

        # Fallback: extract first few sentences
        sentences = re.split(r'[.!?]+', resume_text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        return " ".join(meaningful_sentences[:3]) if meaningful_sentences else "Experience summary not available"

    def process_resume(self, file_content, filename, job_description):
        """Process a single resume and return candidate data"""
        # Extract text
        resume_text = self.extract_text(file_content, filename)
        if resume_text.startswith("Error"):
            print(f"❌ Error processing {filename}: {resume_text}")
            return None

        # Extract basic info
        name, email = self.extract_info_regex(resume_text)

        # Extract skills
        skills = self.extract_skills(resume_text)

        # Calculate score
        score = self.calculate_score(resume_text, job_description)

        # Generate summary
        summary = self.generate_summary(resume_text, job_description)

        # Extract experience highlight
        experience = self._extract_experience_highlight(resume_text)

        # Create Candidate object
        candidate_id = f"{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"

        return Candidate(
            id=candidate_id,
            name=name,
            email=email,
            score=score,
            summary=summary,
            skills=skills,
            experience=experience,
            resume_text=resume_text
        )

    def _extract_experience_highlight(self, text):
        """Extract experience-related text"""
        # Look for experience section
        experience_patterns = [
            r'experience.*?(\n.*?){5}',
            r'work history.*?(\n.*?){5}',
            r'professional.*?(\n.*?){5}'
        ]

        for pattern in experience_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)[:300] + "..."

        # Fallback: take first 200 characters
        return text[:200] + "..."
     

job_title = "AI develpoer"
job_description_text = "We are seeking a skilled AI Developer to design, develop, and deploy artificial intelligence solutions. The ideal candidate will have strong expertise in machine learning, deep learning, and software development, with a passion for creating intelligent systems that solve complex problems."
job_requirements = "Python programming, Machine Learning, Deep Learning, TensorFlow, PyTorch, Natural Language Processing, Data Preprocessing, Cloud platforms, Git version control, API development"

# Create JobDescription object
jd = JobDescription(
    title=job_title,
    description=job_description_text,
    requirements=[req.strip() for req in job_requirements.split(",")]
)

display(HTML(f"""

    🎯 Job Description Loaded
    Title: {jd.title}
    Description: {jd.description}
    Requirements: {', '.join(jd.requirements)}

"""))
     
🎯 Job Description Loaded
Title: AI develpoer

Description: We are seeking a skilled AI Developer to design, develop, and deploy artificial intelligence solutions. The ideal candidate will have strong expertise in machine learning, deep learning, and software development, with a passion for creating intelligent systems that solve complex problems.

Requirements: Python programming, Machine Learning, Deep Learning, TensorFlow, PyTorch, Natural Language Processing, Data Preprocessing, Cloud platforms, Git version control, API development


print("📁 Please upload resume files (PDF or DOCX):")
uploaded_files = files.upload()

if not uploaded_files:
    print("⚠️ Please upload at least one resume file.")
else:
    print(f"✅ Uploaded {len(uploaded_files)} file(s)")

     
📁 Please upload resume files (PDF or DOCX):
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving ANUSREE-K-AI DEVELOPER_RESUME.pdf to ANUSREE-K-AI DEVELOPER_RESUME (1).pdf
✅ Uploaded 1 file(s)

print("🔄 Initializing resume processor...")
processor = ResumeProcessor()
candidates = []

for filename, file_content in uploaded_files.items():
    print(f"🔍 Processing {filename}...")
    candidate = processor.process_resume(file_content, filename, jd)
    if candidate:
        candidates.append(candidate)
        print(f"   ✅ Processed: {candidate.name} - Score: {candidate.score}")
    else:
        print(f"   ❌ Failed to process: {filename}")

# Sort candidates by score
candidates.sort(key=lambda x: x.score, reverse=True)

     
🔄 Initializing resume processor...
🔄 Loading free NLP models...
Device set to use cpu
✅ NLP models initialized
🔍 Processing ANUSREE-K-AI DEVELOPER_RESUME (1).pdf...
   ✅ Processed: Unknown Candidate - Score: 71.5

display(HTML(f"🏆 Ranked Candidates ({len(candidates)} found)"))

if not candidates:
    print("❌ No candidates processed successfully. Check your file formats.")
else:
    for i, candidate in enumerate(candidates, 1):
        display(HTML(f"#{i} - Score: {candidate.score:.1f}/100"))
        candidate.display()

     
🏆 Ranked Candidates (1 found)
#1 - Score: 71.5/100
👤 Unknown Candidate - Score: 71.5/100
📧 Email: kanusreek5@gmail.com

📋 Summary: Anusree Kaunusreek 5@gmail.com 8157825339 Kuttiyil house,Eramala(po),Vatakara(via),673501 . He is an AI developer with expertise in machine learning, deep learning, and natural language processing .

🛠️ Skills: numpy, aws, go, tensorflow, seaborn, python, git, scikit, sql, matplotlib...

💼 Experience Highlight: experience in geographic information systems and disaster management. skills summary frameworks / libraries tensorflow/keras, scikit-learn, opencv, ......


top_n = min(3, len(candidates))  # @param {type:"slider", min:1, max:10, step:1}

if candidates:
    selected_candidates = candidates[:top_n]

    display(HTML(f"""
    
        ✅ Selected Top {top_n} Candidates
    
    """))

    for i, candidate in enumerate(selected_candidates, 1):
        display(HTML(f"#{i} - {candidate.name} ({candidate.score:.1f})"))
        candidate.display()

    # @title ### **11. Generate Interview Schedule**
    display(HTML("📅 Proposed Interview Schedule"))

    # Generate interview times (next business days at 10 AM)
    interview_times = []
    current_date = datetime.now()
    for i in range(len(selected_candidates)):
        # Skip weekends
        while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            current_date += timedelta(days=1)

        interview_time = current_date.replace(hour=10, minute=0, second=0, microsecond=0)
        interview_times.append(interview_time)
        current_date += timedelta(days=1)

    for candidate, interview_time in zip(selected_candidates, interview_times):
        display(HTML(f"""
        
            👤 {candidate.name}
            📧 {candidate.email}
            📅 Proposed Interview: {interview_time.strftime('%A, %B %d, %Y at %I:%M %p')}
            🔗 Calendar integration ready for production use
        
        """))
else:
    print("❌ No candidates to select")

     
✅ Selected Top 1 Candidates
#1 - Unknown Candidate (71.5)
👤 Unknown Candidate - Score: 71.5/100
📧 Email: kanusreek5@gmail.com

📋 Summary: Anusree Kaunusreek 5@gmail.com 8157825339 Kuttiyil house,Eramala(po),Vatakara(via),673501 . He is an AI developer with expertise in machine learning, deep learning, and natural language processing .

🛠️ Skills: numpy, aws, go, tensorflow, seaborn, python, git, scikit, sql, matplotlib...

💼 Experience Highlight: experience in geographic information systems and disaster management. skills summary frameworks / libraries tensorflow/keras, scikit-learn, opencv, ......

📅 Proposed Interview Schedule
👤 Unknown Candidate
📧 kanusreek5@gmail.com

📅 Proposed Interview: Monday, August 25, 2025 at 10:00 AM

🔗 Calendar integration ready for production use


if candidates:
    results = {
        "job_description": jd.to_dict(),
        "processing_date": datetime.now().isoformat(),
        "candidates": [
            {
                "id": c.id,
                "name": c.name,
                "email": c.email,
                "score": c.score,
                "summary": c.summary,
                "skills": c.skills,
                "selected": i < top_n
            }
            for i, c in enumerate(candidates)
        ]
    }

    # Create downloadable file
    json_str = json.dumps(results, indent=2)
    filename = 'candidate_analysis_results.json'
    with open(filename, 'w') as f:
        f.write(json_str)

    files.download(filename)
    print("✅ Results exported as candidate_analysis_results.json")
else:
    print("❌ No results to export")
     
✅ Results exported as candidate_analysis_results.json
