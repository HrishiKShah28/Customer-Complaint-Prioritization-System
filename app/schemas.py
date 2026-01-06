from pydantic import BaseModel
class TextRequest(BaseModel):
    text:str
class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    priority: str
