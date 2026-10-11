from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import json
import os
from pathlib import Path

app = FastAPI()

# Enable CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load model, fallback to offline responses if it fails
model_loaded = False
safety_qa = None

try:
    from transformers import pipeline
    print("Loading safety model... This may take a minute...")
    
    # Use local model path if already downloaded
    model_path = "./model/ConstructionSafetyQA-1.2B-V1"
    if not os.path.exists(model_path):
        # First time - download and save locally
        safety_qa = pipeline(
            "text-generation", 
            model="yasserrmd/ConstructionSafetyQA-1.2B-V1",
            device="cpu"
        )
        # Save for offline use
        safety_qa.save_pretrained(model_path)
    else:
        # Load from local cache - works offline!
        safety_qa = pipeline(
            "text-generation",
            model=model_path,
            device="cpu"
        )
    
    model_loaded = True
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Could not load model: {e}")
    print("Running in offline fallback mode...")

# Load offline responses
with open('offline_responses.json', 'r') as f:
    OFFLINE_RESPONSES = json.load(f)

class SafetyQuestion(BaseModel):
    question: str
    context: str = "General Construction Site"

def get_fallback_response(question: str) -> str:
    """Intelligent fallback when model isn't available"""
    question_lower = question.lower()
    
    # Check for keyword matches
    for key, response in OFFLINE_RESPONSES.items():
        if any(word in question_lower for word in key.split('_')):
            return response
    
    return OFFLINE_RESPONSES.get('default', 
        "Please consult your safety manual or supervisor for guidance on this specific issue.")

@app.get("/")
def read_root():
    return {
        "status": "SafetyBot API Running",
        "mode": "AI Model Active" if model_loaded else "Offline Mode",
        "model_loaded": model_loaded
    }

@app.post("/ask")
async def ask_safety(query: SafetyQuestion):
    
    if model_loaded and safety_qa:
        try:
            prompt = f"Context: {query.context}\nQuestion: {query.question}\nAnswer:"
            
            response = safety_qa(
                prompt, 
                max_length=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256
            )
            
            answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        except Exception as e:
            print(f"Model inference failed: {e}")
            answer = get_fallback_response(query.question)
    else:
        answer = get_fallback_response(query.question)
    
    return {
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "context": query.context,
        "mode": "ai" if model_loaded else "offline"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8191)