from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
rf_model = joblib.load('recommender.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
print("Models loaded!")

# Groq API configuration - reads from Railway environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def get_groq_explanation(crop, confidence, N, P, K, temp, humidity, ph, rainfall):
    """Call Groq API to generate natural language explanation (XAI layer)"""
    
    if not GROQ_API_KEY:
        return get_fallback_explanation(crop, N, P, K, temp, humidity, ph, rainfall)
    
    prompt = f"""You are an agricultural expert. Explain why {crop} is the best crop for these conditions:

Soil:
- Nitrogen: {N} mg/kg
- Phosphorus: {P} mg/kg  
- Potassium: {K} mg/kg
- pH: {ph}

Climate:
- Temperature: {temp}°C
- Humidity: {humidity}%
- Rainfall: {rainfall}mm

The ML model predicted {crop} with {confidence:.1%} confidence.

Provide a practical, farmer-friendly explanation (4-5 sentences) covering:
1. Why this crop matches the soil conditions
2. Why this crop matches the climate conditions
3. One specific growing tip

Keep it concise and helpful."""

    try:
        response = httpx.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=15.0
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return get_fallback_explanation(crop, N, P, K, temp, humidity, ph, rainfall)
            
    except Exception as e:
        print(f"Groq API error: {e}")
        return get_fallback_explanation(crop, N, P, K, temp, humidity, ph, rainfall)


def get_fallback_explanation(crop, N, P, K, temp, humidity, ph, rainfall):
    """Fallback template explanation when Groq API is unavailable"""
    
    fallbacks = {
        'rice': f"Rice is ideal for your conditions. Your soil has {N} nitrogen which is excellent for rice growth. With {rainfall}mm rainfall and {temp}°C temperature, conditions are perfect. Tip: Maintain 5cm standing water during vegetative stage.",
        
        'maize': f"Maize grows well in your conditions. Your balanced soil (N={N}, P={P}, K={K}) supports corn development. Temperature {temp}°C is ideal. Tip: Plant at beginning of rainy season with 75x20cm spacing.",
        
        'chickpea': f"Chickpea is perfect for your conditions. As a legume, it fixes its own nitrogen and improves your soil. Temperature {temp}°C and rainfall {rainfall}mm are ideal. Tip: Do NOT apply nitrogen fertilizer.",
        
        'banana': f"Banana is excellent for your conditions. Your N={N} and K={K} levels are ideal. Temperature {temp}°C and humidity {humidity}% are perfect. Tip: Apply organic mulch around plants.",
        
        'grapes': f"Grapes thrive in your soil. Your high potassium (K={K}) is excellent for grape quality. Temperature {temp}°C is ideal. Tip: Install trellising system for support.",
    }
    
    default = f"{crop.upper()} is recommended for your soil (N={N}, P={P}, K={K}, pH={ph}) and climate ({temp}°C, {humidity}% humidity, {rainfall}mm rainfall)."
    
    return fallbacks.get(crop, default)


@app.get("/")
def root():
    return {"message": "Crop Recommendation API with XAI Layer", "status": "running"}

@app.get("/health")
def health():
    groq_configured = bool(GROQ_API_KEY)
    return {
        "status": "healthy", 
        "models_loaded": True,
        "groq_configured": groq_configured,
        "xai_layer": "active" if groq_configured else "fallback_mode"
    }

@app.post("/predict")
def predict(req: CropRequest):
    # Prepare input
    input_data = np.array([[
        req.N, req.P, req.K,
        req.temperature, req.humidity,
        req.ph, req.rainfall
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    probs = rf_model.predict_proba(input_scaled)[0]
    pred_idx = np.argmax(probs)
    crop = encoder.inverse_transform([pred_idx])[0]
    confidence = float(max(probs))
    
    # Get alternatives
    top_idx = np.argsort(probs)[::-1][1:5]
    alternatives = [encoder.inverse_transform([idx])[0] for idx in top_idx]
    
    # XAI Layer: Generate explanation using Groq
    explanation = get_groq_explanation(
        crop, confidence,
        req.N, req.P, req.K,
        req.temperature, req.humidity,
        req.ph, req.rainfall
    )
    
    return {
        "recommended_crop": crop,
        "confidence": confidence,
        "confidence_percentage": f"{confidence*100:.1f}%",
        "alternatives": alternatives,
        "explanation": explanation,
        "xai_method": "groq_api" if GROQ_API_KEY else "template_fallback"
    }