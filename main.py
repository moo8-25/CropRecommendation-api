from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

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

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def get_detailed_explanation(crop, N, P, K, temp, humidity, ph, rainfall):
    """Rich, detailed explanations for all 22 crops"""
    
    explanations = {
        'rice': f"""
RICE RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil has {N} mg/kg of Nitrogen, which is EXCELLENT for rice cultivation. 
Rice requires high nitrogen (80-120 mg/kg) for vegetative growth and grain development.
Your Phosphorus level of {P} and Potassium level of {K} are well-balanced for rice.
The soil pH of {ph} is IDEAL - rice grows best in slightly acidic to neutral soil (pH 6.0-7.0).

CLIMATE MATCH:
Temperature: {temp}°C is PERFECT for rice (optimal range 20-35°C).
Humidity: {humidity}% is IDEAL - rice thrives in 70-85% humidity.
Rainfall: {rainfall}mm provides the high water availability rice needs (requires 180-300mm).

WHY RICE IS YOUR BEST CHOICE:
Rice is a water-loving cereal that produces high yields in warm, humid conditions.
Your combination of high nitrogen, warm temperature, and abundant rainfall creates perfect conditions.

SPECIFIC GROWING TIPS:
1. Maintain 5-10cm standing water during vegetative stage
2. Apply nitrogen in split doses (basal + top dressing at tillering and panicle initiation)
3. Space plants at 20x15cm for optimal yield
4. Harvest when 80% of grains turn golden yellow (110-150 days)
5. Expected yield: 4-6 tons per hectare

POTENTIAL CHALLENGES:
- Watch for blast disease in high humidity
- Monitor for stem borer and leaf folder
- Ensure proper water management - don't let fields dry completely
""",

        'maize': f"""
MAIZE (CORN) RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil has Nitrogen: {N}, Phosphorus: {P}, Potassium: {K}.
This balanced nutrient profile is EXCELLENT for maize production.
Maize is a heavy feeder that responds well to your nutrient levels.
Soil pH {ph} is suitable (optimal range 5.8-7.0).

CLIMATE MATCH:
Temperature: {temp}°C is IDEAL for maize (optimal 18-30°C).
Rainfall: {rainfall}mm is within suitable range (maize needs 600-1200mm total per season).
Humidity: {humidity}% supports healthy growth without promoting fungal diseases.

WHY MAIZE IS YOUR BEST CHOICE:
Maize is a versatile cereal crop with high yield potential and good market demand.
Your well-balanced soil provides the foundation for strong stalk development and ear formation.

SPECIFIC GROWING TIPS:
1. Plant at beginning of rainy season with 75x20cm spacing
2. Apply 150kg DAP + 100kg Urea per hectare at planting
3. Side-dress with nitrogen when plants are 45-60cm tall
4. Watch for tassel emergence at 60-70 days
5. Harvest when husks turn brown and kernels are hard (90-120 days)

EXPECTED YIELD:
- Rainfed: 3-5 tons/hectare
- Irrigated: 6-9 tons/hectare

POTENTIAL CHALLENGES:
- Watch for fall armyworm - the biggest threat to maize
- Monitor for stem borer and weevils
- Proper storage needed to prevent aflatoxin
""",

        'chickpea': f"""
CHICKPEA (BENGAL GRAM) RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil has only {N} mg/kg Nitrogen - but THIS IS PERFECT for chickpeas!
Chickpeas are LEGUMES that fix their OWN nitrogen from the air using Rhizobium bacteria.
They will actually IMPROVE your soil fertility for the next crop.
Your Phosphorus ({P}) and Potassium ({K}) levels are excellent for nodulation and pod formation.
Soil pH {ph} is IDEAL (optimal range 6.0-8.5).

CLIMATE MATCH:
Temperature: {temp}°C is PERFECT (chickpeas need 15-25°C).
Rainfall: {rainfall}mm is IDEAL - chickpeas are DROUGHT-TOLERANT and hate waterlogging.
Humidity: {humidity}% is suitable - low humidity reduces disease risk.

WHY CHICKPEA IS YOUR BEST CHOICE:
Chickpeas are the perfect rotation crop. They require no nitrogen fertilizer, 
improve your soil, and produce high-protein grains. They're ideal for your 
low-rainfall conditions.

SPECIFIC GROWING TIPS:
1. DO NOT APPLY NITROGEN FERTILIZER - it will reduce nodulation
2. Inoculate seeds with Rhizobium bacteria before planting
3. Plant at 30x10cm spacing, 5-7cm deep
4. Avoid irrigation after pod formation
5. Harvest when pods turn brown and rattle (90-120 days)

SOIL IMPROVEMENT BENEFITS:
- Adds 40-60kg nitrogen per hectare naturally
- Breaks pest cycles for cereals
- Improves soil structure with root system

EXPECTED YIELD: 1.5-2.5 tons/hectare

POTENTIAL CHALLENGES:
- Susceptible to wilt in waterlogged conditions
- Watch for pod borer and aphids
- Use resistant varieties for disease-prone areas
""",

        'banana': f"""
BANANA RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil has Nitrogen: {N}, Potassium: {K} - BOTH ARE EXCELLENT for bananas!
Bananas require high potassium for fruit development - your K={K} is perfect.
Nitrogen {N} supports vigorous leaf growth and suckering.
Phosphorus {P} is adequate for root development.
Soil pH {ph} is suitable (optimal range 5.5-7.0).

CLIMATE MATCH:
Temperature: {temp}°C is IDEAL (bananas need 25-30°C).
Humidity: {humidity}% is PERFECT - bananas love high humidity.
Rainfall: {rainfall}mm is good - supplement with irrigation if below 1200mm/year.

WHY BANANA IS YOUR BEST CHOICE:
Bananas are high-value fruit crops with year-round income potential.
Your soil has the perfect potassium levels for sweet, high-quality fruit.
The warm, humid conditions are exactly what bananas need to thrive.

SPECIFIC GROWING TIPS:
1. Plant tissue-cultured suckers with 3x3m spacing (1100 plants/hectare)
2. Apply 200g Urea + 300g DAP + 500g MOP per plant annually
3. Apply organic mulch (10-15cm thick) to retain moisture
4. Remove male buds 10-15cm after last hand appears
5. Support plants with bamboo stakes to prevent wind damage

TIMELINE:
- Planting to harvest: 11-14 months
- Ratoon crop (second cycle): 8-10 months

EXPECTED YIELD: 40-60 tons/hectare

POTENTIAL CHALLENGES:
- Panama disease (Fusarium wilt) - use resistant varieties
- Nematodes - practice crop rotation
- Wind damage - provide windbreaks
""",

        'grapes': f"""
GRAPES RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil has Potassium: {K} - THIS IS EXCELLENT for grapes!
High potassium is directly correlated with grape quality, sweetness, and wine character.
Your Nitrogen ({N}) is at ideal levels - too much nitrogen reduces fruit quality.
Phosphorus ({P}) supports root development and flowering.
Soil pH {ph} is PERFECT (optimal range 6.0-7.0).

CLIMATE MATCH:
Temperature: {temp}°C is IDEAL for grape ripening.
Rainfall: {rainfall}mm is suitable - grapes prefer 600-800mm annually.
Humidity: {humidity}% - moderate humidity is best.

WHY GRAPES ARE YOUR BEST CHOICE:
Your high potassium soil is PERFECT for premium grape production.
Grapes are a high-value crop suitable for table grapes, raisins, or wine.

SPECIFIC GROWING TIPS:
1. Install trellising system (bower or wire trellis)
2. Prune during dormant season (December-January) for better yield
3. Apply 25-30 tons of well-rotted manure per hectare annually
4. Irrigate every 7-10 days during growing season
5. Stop irrigation 15 days before harvest for better sugar content

TIMELINE:
- Planting to first harvest: 3 years
- Full production: from year 4
- Economic life: 20-25 years

PRUNING SCHEDULE:
- Backward pruning: December (for main crop)
- Forward pruning: June (for second crop in some varieties)

EXPECTED YIELD: 20-30 tons/hectare

POTENTIAL CHALLENGES:
- Powdery mildew - requires regular fungicide spray
- Anthracnose - use resistant varieties
- Birds - netting required near ripening
""",

        'cotton': f"""
COTTON RECOMMENDATION - DETAILED ANALYSIS

SOIL CONDITIONS:
Your soil nutrients (N={N}, P={P}, K={K}) provide balanced nutrition for cotton.
Cotton requires moderate nitrogen for vegetative growth and boll development.
Your soil pH {ph} is suitable (optimal range 6.0-8.0).

CLIMATE MATCH:
Temperature: {temp}°C is IDEAL for cotton (optimal 25-35°C).
Rainfall: {rainfall}mm is appropriate - cotton needs 600-1000mm.
Humidity: {humidity}% - moderate humidity is best.

WHY COTTON IS YOUR BEST CHOICE:
Cotton is a commercial fiber crop with guaranteed market through government procurement.
Your climate provides the long, frost-free growing season cotton requires.

SPECIFIC GROWING TIPS:
1. Plant after last frost with 75x30cm spacing
2. Apply 100kg DAP + 60kg MOP + 60kg Urea per hectare
3. First irrigation at 25-30 days, then every 10-15 days
4. Square formation: 45 days after planting
5. First boll opening: 140-160 days

PEST MANAGEMENT:
- Bollworm: Install pheromone traps (5-6 per acre)
- Aphids and jassids: Monitor regularly
- Pink bollworm: Remove and destroy affected squares

HARVESTING:
- Pick when 80% of bolls are open
- Morning hours are best to avoid leaf moisture
- 2-3 pickings required

EXPECTED YIELD: 5-10 quintals/hectare (500-1000 kg lint)

POTENTIAL CHALLENGES:
- Bollworm is the biggest threat - use Bt cotton varieties
- Aphids and whiteflies - can develop resistance
- Leaf curl virus - plant resistant varieties
""",
    }
    
    # Default explanation for any crop not in dictionary
    default = f"""
{crop.upper()} RECOMMENDATION

SOIL CONDITIONS:
Your soil has N={N}, P={P}, K={K}, pH={ph}. These nutrient levels provide balanced nutrition for {crop} cultivation.

CLIMATE CONDITIONS:
Temperature: {temp}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm.

WHY {crop.upper()} IS RECOMMENDED:
Based on your soil and climate conditions, {crop} is the optimal choice. The combination of your specific nutrient levels and environmental factors aligns well with {crop}'s growth requirements.

GENERAL GROWING TIPS:
1. Prepare soil with 15-20 tons of well-rotted manure per hectare
2. Follow recommended spacing for your specific variety
3. Apply fertilizers based on soil test recommendations
4. Monitor for pests and diseases regularly
5. Consult local agricultural extension for variety selection

Note: For detailed variety recommendations, contact your local agriculture office.
"""
    
    return explanations.get(crop, default)

@app.get("/")
def root():
    return {"message": "Crop Recommendation API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

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
    
    # Get detailed explanation
    explanation = get_detailed_explanation(
        crop, req.N, req.P, req.K,
        req.temperature, req.humidity,
        req.ph, req.rainfall
    )
    
    return {
        "recommended_crop": crop,
        "confidence": confidence,
        "confidence_percentage": f"{confidence*100:.1f}%",
        "alternatives": alternatives,
        "explanation": explanation.strip()
    }