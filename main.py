from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import json
import urllib.request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Product Transparency AI Service",
    description="AI-powered service for generating intelligent questions and analyzing product transparency",
    version="1.0.0"
)

# CORS middleware
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# Allow CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI service is running"}

@app.post("/generate-questions")
async def generate_questions(request: Request):
    data = await request.json()
    product_data = data.get("productData", {})

    name = product_data.get("name", "")
    category = product_data.get("category", "")
    brand = product_data.get("brand", "")
    ingredients = product_data.get("ingredients", [])
    description = product_data.get("description", "")

    # Simulate AI question generation
    questions = [
        f"What are the benefits of {name} by {brand}?",
        f"Is {name} suitable for all skin types?",
        f"How do ingredients like {', '.join(ingredients)} affect the product?",
        f"What makes this {category.lower()} product unique?",
    ]

    return {"questions": questions}


# Pydantic models
class ProductData(BaseModel):
    name: str
    category: str
    brand: str
    ingredients: List[str]
    description: Optional[str] = None
    certifications: Optional[List[str]] = None
    packaging: Optional[str] = None
    sustainability: Optional[str] = None

class QuestionRequest(BaseModel):
    productData: ProductData

class QuestionResponse(BaseModel):
    id: str
    question: str
    type: str
    options: Optional[List[str]] = None
    required: bool
    order: int

class ScoreRequest(BaseModel):
    productData: ProductData
    answers: Dict[str, Any]

class ScoreResponse(BaseModel):
    score: int
    breakdown: Dict[str, int]
    category: str

class AnalysisRequest(BaseModel):
    product: ProductData
    answers: Dict[str, Any]

class AnalysisResponse(BaseModel):
    summary: str
    transparencyScore: int
    analysis: Dict[str, List[str]]


# AI Service class
class TransparencyAI:
    def __init__(self):
        self.tng_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("TNG_API_KEY")
        self.use_tng = bool(self.tng_api_key)

    def generate_questions(self, product_data: ProductData) -> List[QuestionResponse]:
        """Generate intelligent questions based on product data"""
        try:
            if self.use_tng:
                return self._generate_questions_with_tng(product_data)
            return self._generate_fallback_questions(product_data)
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._generate_fallback_questions(product_data)

    def _generate_questions_with_tng(self, product_data: ProductData) -> List[QuestionResponse]:
        """Generate questions using TNG: DeepSeek R1T2 Chimera API (with proper error handling)"""
        import urllib.request
        import json

        try:
            api_key = self.tng_api_key
            tng_url = os.getenv('DEEPSEEK_API_URL') or os.getenv('TNG_API_URL') or 'https://api.deepseek.ai/v1/generate'

            prompt = f"""
            Generate 5-7 intelligent follow-up questions for a product transparency report.

            Product: {product_data.name}
            Category: {product_data.category}
            Brand: {product_data.brand}
            Ingredients: {', '.join(product_data.ingredients)}

            Focus on:
            - Ingredient sourcing and origin
            - Sustainability practices
            - Certifications and standards
            - Ethical practices
            - Environmental impact
            - Social responsibility

            Return as a JSON array with fields: id, question, type, options (if applicable), required, order
            """

            body = {
                "model": os.getenv("DEEPSEEK_MODEL", "chimera-r1t2"),
                "prompt": prompt,
                "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "800"))
            }

            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                tng_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_text = resp.read().decode("utf-8")
                resp_json = json.loads(resp_text)

            # DEBUG: log the raw API response
            logger.info(f"TNG raw response: {json.dumps(resp_json, indent=2)}")

            # Try to parse typical response shapes
            choices = resp_json.get("choices") or resp_json.get("outputs") or resp_json.get("data") or []
            content = ""
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                content = first.get("text") or first.get("content") or first.get("output") or ""
            else:
                content = resp_json.get("text") or resp_json.get("output") or ""

            # Attempt to parse JSON array from content
            questions: List[QuestionResponse] = []
            try:
                parsed = json.loads(content)
                for item in parsed:
                    questions.append(QuestionResponse(
                        id=item.get("id", f"tng_{len(questions)}"),
                        question=item.get("question") or item.get("text") or "",
                        type=item.get("type", "text"),
                        options=item.get("options"),
                        required=bool(item.get("required", False)),
                        order=int(item.get("order", 0))
                    ))
                if questions:
                    logger.info(f"TNG parsed {len(questions)} questions successfully")
                    return questions
            except Exception:
                # Fallback if content is not valid JSON
                logger.warning("TNG content is not valid JSON, using line-based fallback")

            # Fallback: split content into lines
            lines = [ln.strip() for ln in (content or "").split("\n") if ln.strip()]
            for i, line in enumerate(lines[:7]):
                questions.append(QuestionResponse(
                    id=f"tng_{i}",
                    question=line[:1000],
                    type="text",
                    required=False,
                    order=i
                ))

            if questions:
                logger.info(f"TNG fallback generated {len(questions)} questions from text lines")
                return questions

        # Final fallback: built-in questions
            logger.warning("TNG returned empty response, using built-in fallback questions")
            return self._generate_fallback_questions(product_data)

        except Exception as e:
            logger.error(f"TNG/DeepSeek API error: {e}")
            return self._generate_fallback_questions(product_data)




    def _generate_fallback_questions(self, product_data: ProductData) -> List[QuestionResponse]:
        """Fallback questions if AI fails"""
        base_questions = [
            QuestionResponse(id="sourcing", question="What is the source and origin of your primary ingredients?", type="text", required=True, order=0),
            QuestionResponse(id="certifications", question="What certifications does your product have?", type="multiselect",
                             options=["Organic", "Fair Trade", "Cruelty-Free", "Vegan", "Non-GMO", "USDA Certified", "None"], required=False, order=1),
            QuestionResponse(id="packaging", question="What type of packaging do you use?", type="select",
                             options=["Recyclable", "Biodegradable", "Compostable", "Reusable", "Standard Plastic", "Other"], required=True, order=2),
            QuestionResponse(id="sustainability", question="What sustainability practices does your company follow?", type="text", required=False, order=3),
            QuestionResponse(id="testing", question="Do you test on animals?", type="select",
                             options=["No, never", "No, but parent company does", "Yes, for safety", "Yes, for efficacy", "Prefer not to say"], required=True, order=4)
        ]

        if product_data.category == "Food & Beverage":
            base_questions.append(QuestionResponse(id="allergens", question="What allergens are present in your product?", type="multiselect",
                                                   options=["Nuts", "Dairy", "Gluten", "Soy", "Eggs", "None"], required=True, order=5))

        if product_data.category in ["Skincare", "Personal Care"]:
            base_questions.append(QuestionResponse(id="skin_types", question="What skin types is this product suitable for?", type="multiselect",
                                                   options=["Sensitive", "Dry", "Oily", "Combination", "Normal", "All Types"], required=True, order=5))

        return base_questions

    # Transparency scoring and analysis remain unchanged
    def calculate_transparency_score(self, product_data: ProductData, answers: Dict[str, Any]) -> ScoreResponse:
        score = 0
        breakdown = {}
        if product_data.name: score += 10; breakdown["basic_info"]=10
        if product_data.ingredients: score += 15; breakdown["ingredients"]=15
        if product_data.description: score += 10; breakdown["description"]=10
        if answers.get("sourcing") and len(str(answers["sourcing"])) > 10: score+=20; breakdown["sourcing"]=20
        if answers.get("certifications") and len(answers["certifications"])>0: score+=15; breakdown["certifications"]=15
        if answers.get("packaging") in ["Recyclable","Biodegradable","Compostable"]: score+=15; breakdown["packaging"]=15
        if answers.get("sustainability") and len(str(answers["sustainability"]))>20: score+=15; breakdown["sustainability"]=15
        if answers.get("testing")=="No, never": score+=10; breakdown["ethics"]=10

        if score>=80: category="Excellent"
        elif score>=60: category="Good"
        elif score>=40: category="Fair"
        else: category="Needs Improvement"

        return ScoreResponse(score=min(score,100), breakdown=breakdown, category=category)

    def analyze_product(self, product: ProductData, answers: Dict[str, Any]) -> AnalysisResponse:
        score_response = self.calculate_transparency_score(product, answers)
        strengths = self._generate_strengths(answers)
        improvements = self._generate_improvements(answers)
        recommendations = self._generate_recommendations(answers)
        summary = self._generate_summary(product, score_response, strengths, improvements)
        return AnalysisResponse(summary=summary, transparencyScore=score_response.score, analysis={
            "strengths": strengths,
            "improvements": improvements,
            "recommendations": recommendations
        })

    def _generate_strengths(self, answers: Dict[str, Any]) -> List[str]:
        strengths=[]
        if answers.get("sourcing"): strengths.append("Clear ingredient sourcing information provided")
        if answers.get("certifications") and len(answers["certifications"])>0: strengths.append("Product certifications disclosed")
        if answers.get("packaging") in ["Recyclable","Biodegradable","Compostable"]: strengths.append("Eco-friendly packaging used")
        if answers.get("testing")=="No, never": strengths.append("Cruelty-free practices confirmed")
        if answers.get("sustainability"): strengths.append("Sustainability practices documented")
        if not strengths: strengths.append("Basic product information provided")
        return strengths

    def _generate_improvements(self, answers: Dict[str, Any]) -> List[str]:
        improvements=[]
        if not answers.get("sourcing"): improvements.append("Provide detailed ingredient sourcing information")
        if not answers.get("certifications") or len(answers.get("certifications",[]))==0: improvements.append("Consider obtaining relevant certifications")
        if not answers.get("sustainability"): improvements.append("Add sustainability metrics and practices")
        if answers.get("packaging") not in ["Recyclable","Biodegradable","Compostable"]: improvements.append("Consider more sustainable packaging options")
        if answers.get("testing")!="No, never": improvements.append("Consider transitioning to cruelty-free practices")
        return improvements

    def _generate_recommendations(self, answers: Dict[str, Any]) -> List[str]:
        recommendations=[
            "Consider third-party transparency certifications",
            "Provide detailed ingredient sourcing information",
            "Add sustainability metrics and environmental impact data"
        ]
        if answers.get("testing")!="No, never": recommendations.append("Consider transitioning to cruelty-free practices")
        return recommendations

    def _generate_summary(self, product: ProductData, score_response: ScoreResponse, strengths: List[str], improvements: List[str]) -> str:
        score=score_response.score
        category=score_response.category
        if score>=80: level="excellent"
        elif score>=60: level="good"
        elif score>=40: level="moderate"
        else: level="limited"
        summary=f"Analysis of {product.name}: This product demonstrates {level} transparency with a score of {score}/100 ({category}). "
        if len(strengths)>0: summary+=f"Key strengths include {', '.join(strengths[:2])}. "
        if len(improvements)>0: summary+=f"Areas for improvement include {', '.join(improvements[:2])}."
        return summary


# Initialize AI service
ai_service = TransparencyAI()


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Product Transparency AI Service", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_service": "operational"}

@app.post("/api/ai/generate-questions", response_model=List[QuestionResponse])
async def generate_questions(request: QuestionRequest):
    try:
        questions = ai_service.generate_questions(request.productData)
        return questions
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions")


@app.post("/api/ai/transparency-score", response_model=ScoreResponse)
async def calculate_score(request: ScoreRequest):
    try:
        score = ai_service.calculate_transparency_score(request.productData, request.answers)
        return score
    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate score")

@app.post("/api/ai/analyze-product", response_model=AnalysisResponse)
async def analyze_product(request: AnalysisRequest):
    try:
        analysis = ai_service.analyze_product(request.product, request.answers)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing product: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze product")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
