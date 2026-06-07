from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Crop Recommendation API with AI Explanations")

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD ML MODEL ============
print("Loading ML models...")
rf_model = joblib.load('recommender.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
print("ML models loaded!")

# ============ LOAD LLM ============
print("Loading LLM (FLAN-T5-large)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
if device == "cuda":
    llm = llm.cuda()
llm.eval()
print(f"LLM loaded on {device}!")

# ============ REQUEST/RESPONSE MODELS ============
class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropResponse(BaseModel):
    recommended_crop: str
    confidence: float
    confidence_percentage: str
    alternatives: list
    explanation: str

# ============ TEMPLATE EXPLANATIONS (FALLBACK) ============
def get_template_explanation(crop, N, P, K, temp, humidity, ph, rainfall):
    templates = {
        'rice': f"RICE is ideal for your conditions. Your soil has {N} nitrogen which is excellent for rice growth. The pH of {ph} is perfect. With {rainfall}mm rainfall and {temp}°C, you have the high water availability and warmth rice needs. Tip: Maintain 5cm standing water during vegetative stage.",
        
        'maize': f"MAIZE grows well in your conditions. Your balanced soil (N={N}, P={P}, K={K}) supports corn development. Temperature {temp}°C is ideal for maize. Tip: Plant at beginning of rainy season with 75x20cm spacing.",
        
        'chickpea': f"CHICKPEA is perfect for your conditions. As a legume, it fixes its own nitrogen - improving your soil. Temperature {temp}°C and rainfall {rainfall}mm are ideal. Tip: Do NOT apply nitrogen fertilizer. Inoculate seeds with Rhizobium bacteria.",
        
        'banana': f"BANANA is excellent for your conditions. Your N={N} and K={K} levels are ideal. Temperature {temp}°C and humidity {humidity}% are perfect. Tip: Plant suckers with 3x3m spacing and apply organic mulch.",
        
        'grapes': f"GRAPES thrive in your soil. Your high potassium (K={K}) is excellent for grape quality. Temperature {temp}°C is ideal for ripening. Tip: Install trellising system and prune during dormant season.",
        
        'cotton': f"COTTON suits your climate. Temperature {temp}°C and rainfall {rainfall}mm are appropriate. Tip: Plant after last frost with 75x30cm spacing.",
        
        'mango': f"MANGO thrives in your warm conditions. Temperature {temp}°C is ideal. Your soil pH {ph} is suitable. Tip: Plant in well-drained soil during early rainy season.",
        
        'orange': f"ORANGE grows well in your conditions. Temperature {temp}°C and rainfall {rainfall}mm are suitable. Your soil provides good nutrition. Tip: Ensure good drainage and protect from frost.",
        
        'papaya': f"PAPAYA is well-suited for your conditions. Warm temperature {temp}°C and your soil nutrients support growth. Tip: Plant in sunny location with well-drained soil.",
        
        'watermelon': f"WATERMELON thrives in your warm conditions. Temperature {temp}°C and rainfall {rainfall}mm are ideal. Tip: Plant in sandy, well-drained soil.",
        
        'muskmelon': f"MUSKMELON grows well in your conditions. Warm temperature {temp}°C and moderate rainfall suit this crop. Tip: Harvest when fruit separates easily from vine.",
        
        'pomegranate': f"POMEGRANATE suits your conditions. It tolerates your soil pH and temperature. Tip: Prune to maintain shape and improve air circulation.",
        
        'coconut': f"COCONUT thrives in tropical conditions. Your temperature and humidity are ideal. Tip: Plant in well-drained sandy loam soil.",
        
        'apple': f"APPLE requires specific conditions. Your temperature is suitable for certain varieties. Tip: Choose appropriate variety for your climate zone.",
        
        'coffee': f"COFFEE grows well in your conditions. Temperature {temp}°C and rainfall {rainfall}mm are suitable. Tip: Provide shade for young plants.",
        
        'jute': f"JUTE thrives in warm, humid conditions with good rainfall. Your conditions are favorable. Tip: Harvest when flowers appear for best fiber quality.",
        
        'lentil': f"LENTIL is well-suited for your conditions. As a legume, it improves soil fertility. Tip: Do not overwater - lentils prefer drier conditions.",
        
        'mothbeans': f"MOTHBEANS grow well in your conditions. This drought-tolerant legume fixes nitrogen. Tip: Suitable for dryland farming.",
        
        'mungbean': f"MUNGBEAN is a good choice. This short-duration legume fixes nitrogen. Tip: Harvest when 90% of pods turn black.",
        
        'blackgram': f"BLACKGRAM suits your conditions. This heat-tolerant legume improves soil. Tip: Avoid waterlogging - requires well-drained soil.",
        
        'pigeonpeas': f"PIGEONPEAS are excellent for your conditions. This perennial legume fixes nitrogen and tolerates drought. Tip: Can be intercropped with cereals."
    }
    
    default = f"{crop.upper()} is recommended for your conditions. Your soil (N={N}, P={P}, K={K}, pH={ph}) and climate ({temp}°C, {humidity}% humidity, {rainfall}mm rainfall) are suitable. Tip: Prepare soil with organic matter and monitor for pests regularly."
    
    return templates.get(crop, default)

# ============ LLM EXPLANATION FUNCTION ============
def get_llm_explanation(crop, confidence, N, P, K, temp, humidity, ph, rainfall):
    prompt = f"""Explain why {crop} is the best crop for these conditions:

Soil: N={N}, P={P}, K={K}, pH={ph}
Climate: {temp}°C, {humidity}% humidity, {rainfall}mm rainfall

Provide a practical 3-4 sentence explanation for a farmer. Include one specific growing tip."""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
    
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Fallback to template if LLM response is too short
    if len(explanation) < 50:
        explanation = get_template_explanation(crop, N, P, K, temp, humidity, ph, rainfall)
    
    return explanation

# ============ API ENDPOINTS ============
@app.get("/")
def root():
    return {"message": "Crop Recommendation API with AI Explanations", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_model_loaded": True,
        "llm_loaded": True,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict", response_model=CropResponse)
def predict(request: CropRequest):
    # ML Prediction
    input_data = np.array([[
        request.N, request.P, request.K,
        request.temperature, request.humidity,
        request.ph, request.rainfall
    ]])
    
    input_scaled = scaler.transform(input_data)
    probs = rf_model.predict_proba(input_scaled)[0]
    pred_idx = np.argmax(probs)
    crop = encoder.inverse_transform([pred_idx])[0]
    confidence = float(max(probs))
    
    # Get alternatives
    top_idx = np.argsort(probs)[::-1][1:4]
    alternatives = [encoder.inverse_transform([idx])[0] for idx in top_idx]
    
    # Get LLM Explanation
    explanation = get_llm_explanation(
        crop, confidence,
        request.N, request.P, request.K,
        request.temperature, request.humidity,
        request.ph, request.rainfall
    )
    
    return CropResponse(
        recommended_crop=crop,
        confidence=confidence,
        confidence_percentage=f"{confidence*100:.1f}%",
        alternatives=alternatives,
        explanation=explanation
    )

# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)