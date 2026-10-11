from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

model = None
tokenizer = None

# NLLB language codes mapping
LANGUAGE_PAIRS = {
    # English to other languages
    "en-ar": {"source": "English", "target": "Arabic", "src_code": "eng_Latn", "tgt_code": "arb_Arab"},
    "en-cs": {"source": "English", "target": "Czech", "src_code": "eng_Latn", "tgt_code": "ces_Latn"},
    "en-da": {"source": "English", "target": "Danish", "src_code": "eng_Latn", "tgt_code": "dan_Latn"},
    "en-de": {"source": "English", "target": "German", "src_code": "eng_Latn", "tgt_code": "deu_Latn"},
    "en-es": {"source": "English", "target": "Spanish", "src_code": "eng_Latn", "tgt_code": "spa_Latn"},
    "en-fi": {"source": "English", "target": "Finnish", "src_code": "eng_Latn", "tgt_code": "fin_Latn"},
    "en-fr": {"source": "English", "target": "French", "src_code": "eng_Latn", "tgt_code": "fra_Latn"},
    "en-hr": {"source": "English", "target": "Croatian", "src_code": "eng_Latn", "tgt_code": "hrv_Latn"},
    "en-hu": {"source": "English", "target": "Hungarian", "src_code": "eng_Latn", "tgt_code": "hun_Latn"},
    "en-id": {"source": "English", "target": "Indonesian", "src_code": "eng_Latn", "tgt_code": "ind_Latn"},
    "en-it": {"source": "English", "target": "Italian", "src_code": "eng_Latn", "tgt_code": "ita_Latn"},
    "en-ja": {"source": "English", "target": "Japanese", "src_code": "eng_Latn", "tgt_code": "jpn_Jpan"},
    "en-ko": {"source": "English", "target": "Korean", "src_code": "eng_Latn", "tgt_code": "kor_Hang"},
    "en-ms": {"source": "English", "target": "Malay", "src_code": "eng_Latn", "tgt_code": "zsm_Latn"},
    "en-nb": {"source": "English", "target": "Norwegian Bokmal", "src_code": "eng_Latn", "tgt_code": "nob_Latn"},
    "en-nl": {"source": "English", "target": "Dutch", "src_code": "eng_Latn", "tgt_code": "nld_Latn"},
    "en-no": {"source": "English", "target": "Norwegian", "src_code": "eng_Latn", "tgt_code": "nno_Latn"},
    "en-pl": {"source": "English", "target": "Polish", "src_code": "eng_Latn", "tgt_code": "pol_Latn"},
    "en-pt": {"source": "English", "target": "Portuguese", "src_code": "eng_Latn", "tgt_code": "por_Latn"},
    "en-ro": {"source": "English", "target": "Romanian", "src_code": "eng_Latn", "tgt_code": "ron_Latn"},
    "en-ru": {"source": "English", "target": "Russian", "src_code": "eng_Latn", "tgt_code": "rus_Cyrl"},
    "en-sv": {"source": "English", "target": "Swedish", "src_code": "eng_Latn", "tgt_code": "swe_Latn"},
    "en-th": {"source": "English", "target": "Thai", "src_code": "eng_Latn", "tgt_code": "tha_Thai"},
    "en-tr": {"source": "English", "target": "Turkish", "src_code": "eng_Latn", "tgt_code": "tur_Latn"},
    "en-uk": {"source": "English", "target": "Ukrainian", "src_code": "eng_Latn", "tgt_code": "ukr_Cyrl"},
    "en-vi": {"source": "English", "target": "Vietnamese", "src_code": "eng_Latn", "tgt_code": "vie_Latn"},
    "en-zh": {"source": "English", "target": "Chinese", "src_code": "eng_Latn", "tgt_code": "zho_Hans"},
    
    # Other languages to English
    "ar-en": {"source": "Arabic", "target": "English", "src_code": "arb_Arab", "tgt_code": "eng_Latn"},
    "cs-en": {"source": "Czech", "target": "English", "src_code": "ces_Latn", "tgt_code": "eng_Latn"},
    "da-en": {"source": "Danish", "target": "English", "src_code": "dan_Latn", "tgt_code": "eng_Latn"},
    "de-en": {"source": "German", "target": "English", "src_code": "deu_Latn", "tgt_code": "eng_Latn"},
    "es-en": {"source": "Spanish", "target": "English", "src_code": "spa_Latn", "tgt_code": "eng_Latn"},
    "fi-en": {"source": "Finnish", "target": "English", "src_code": "fin_Latn", "tgt_code": "eng_Latn"},
    "fr-en": {"source": "French", "target": "English", "src_code": "fra_Latn", "tgt_code": "eng_Latn"},
    "hr-en": {"source": "Croatian", "target": "English", "src_code": "hrv_Latn", "tgt_code": "eng_Latn"},
    "hu-en": {"source": "Hungarian", "target": "English", "src_code": "hun_Latn", "tgt_code": "eng_Latn"},
    "id-en": {"source": "Indonesian", "target": "English", "src_code": "ind_Latn", "tgt_code": "eng_Latn"},
    "it-en": {"source": "Italian", "target": "English", "src_code": "ita_Latn", "tgt_code": "eng_Latn"},
    "ja-en": {"source": "Japanese", "target": "English", "src_code": "jpn_Jpan", "tgt_code": "eng_Latn"},
    "ko-en": {"source": "Korean", "target": "English", "src_code": "kor_Hang", "tgt_code": "eng_Latn"},
    "ms-en": {"source": "Malay", "target": "English", "src_code": "zsm_Latn", "tgt_code": "eng_Latn"},
    "nb-en": {"source": "Norwegian Bokmal", "target": "English", "src_code": "nob_Latn", "tgt_code": "eng_Latn"},
    "nl-en": {"source": "Dutch", "target": "English", "src_code": "nld_Latn", "tgt_code": "eng_Latn"},
    "no-en": {"source": "Norwegian", "target": "English", "src_code": "nno_Latn", "tgt_code": "eng_Latn"},
    "pl-en": {"source": "Polish", "target": "English", "src_code": "pol_Latn", "tgt_code": "eng_Latn"},
    "pt-en": {"source": "Portuguese", "target": "English", "src_code": "por_Latn", "tgt_code": "eng_Latn"},
    "ro-en": {"source": "Romanian", "target": "English", "src_code": "ron_Latn", "tgt_code": "eng_Latn"},
    "ru-en": {"source": "Russian", "target": "English", "src_code": "rus_Cyrl", "tgt_code": "eng_Latn"},
    "sv-en": {"source": "Swedish", "target": "English", "src_code": "swe_Latn", "tgt_code": "eng_Latn"},
    "th-en": {"source": "Thai", "target": "English", "src_code": "tha_Thai", "tgt_code": "eng_Latn"},
    "tr-en": {"source": "Turkish", "target": "English", "src_code": "tur_Latn", "tgt_code": "eng_Latn"},
    "uk-en": {"source": "Ukrainian", "target": "English", "src_code": "ukr_Cyrl", "tgt_code": "eng_Latn"},
    "vi-en": {"source": "Vietnamese", "target": "English", "src_code": "vie_Latn", "tgt_code": "eng_Latn"},
    "zh-en": {"source": "Chinese", "target": "English", "src_code": "zho_Hans", "tgt_code": "eng_Latn"},
}

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Translation API is active",
        "model_loaded": model is not None,
        "model": MODEL_NAME
    }

@app.post("/load_models")
async def load_models():
    global model, tokenizer
    
    if model is not None:
        return {"status": "already_loaded", "message": "Models are already loaded"}
    
    try:
        logger.info("Loading NLLB tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info("Loading NLLB translation model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        logger.info("Models loaded successfully!")
        return {
            "status": "success",
            "message": "Translation models loaded successfully",
            "model": MODEL_NAME,
            "info": "NLLB-200 supports 200 languages with high quality"
        }
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=400,
            detail="Models not loaded. Please call /load_models first."
        )
    
    lang_pair = f"{request.source_lang}-{request.target_lang}"
    if lang_pair not in LANGUAGE_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language pair: {lang_pair}"
        )
    
    try:
        pair_info = LANGUAGE_PAIRS[lang_pair]
        
        # Set source language for tokenizer
        tokenizer.src_lang = pair_info['src_code']
        
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate translation with target language code
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(pair_info['tgt_code']),
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode translation
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return {
            "status": "success",
            "original_text": request.text,
            "translated_text": translation,
            "source_language": pair_info['source'],
            "target_language": pair_info['target']
        }
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/language_pairs")
async def get_language_pairs():
    return {
        "available_pairs": {k: {"source": v["source"], "target": v["target"]} for k, v in LANGUAGE_PAIRS.items()},
        "count": len(LANGUAGE_PAIRS)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8192)