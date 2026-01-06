import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "HrishiShah/customer-complaint-priority-bert"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")

LABELS = ["negative", "neutral", "positive"]

def assign_priority(sentiment, confidence):
    if sentiment == "negative" and confidence >= 0.8:
        return "high"
    elif sentiment == "negative":
        return "low"
    else:
        return "low"
def predict_batch(texts, batch_size=32):
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            truncation=True,
            max_length=64,
            padding=True,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        confidences, pred_ids = torch.max(probs, dim=1)

        for c, p in zip(confidences, pred_ids):
            sentiment = LABELS[p.item()]
            confidence = c.item()
            results.append({
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "priority": assign_priority(sentiment, confidence)
            })

    return results

def predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    confidence_tensor, pred_id_tensor = torch.max(probs, dim=1)

    confidence = confidence_tensor.item()
    pred_id = pred_id_tensor.item()
    sentiment = LABELS[pred_id]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "priority": assign_priority(sentiment, confidence)
    }
