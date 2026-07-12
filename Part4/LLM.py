import os
import requests
import json
import joblib
import pandas as pd
from jsonschema import validate, ValidationError
import re
#1.Set up the LLM API connection:
LLM_API_KEY = os.environ["LLM_API_KEY"]
print("✅ API key received")
if not LLM_API_KEY:
    raise ValueError("Missing API key. Set LLM_API_KEY environment variable.")
   

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",  # passkey retrieved securely
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Error:", response.status_code)
        return None
    return response.json()["choices"][0]["message"]["content"]

system_prompt = "You are a structured data extractor. Reply only with plain text."
user_prompt = "Reply with only the word: hello"

output = call_llm(system_prompt, user_prompt)
print("LLM output:", output)

#2. Prompt design:
# Zero-shot system prompt for Track C
system_prompt = """
You are an assistant that explains machine learning model predictions.
Given feature values, a predicted class, and a predicted probability, output only valid JSON.
The JSON must contain exactly these fields:
{
  "prediction_label": "string",
  "confidence_level": "low|medium|high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}
Confidence rubric:
- probability <0.6 → low
- 0.6–0.8 → medium
- >0.8 → high
Do not include any text outside of the JSON object.
"""
#Temperature A/B comparison.
# Example feature-vector inputs (3 test cases)
test_inputs = [
    {"features": {"defect_ratio": 0.50, "severity": 2}, "prediction_label": "1", "prediction_probability": 0.87},
    {"features": {"defect_ratio": 0.05, "severity": 0}, "prediction_label": "0", "prediction_probability": 0.42},
    {"features": {"defect_ratio": 0.30, "severity": 1}, "prediction_label": "0", "prediction_probability": 0.65}
]

# Run comparison for each input
results = []
for inp in test_inputs:
    user_prompt = json.dumps(inp)
    out_temp0 = call_llm(system_prompt, user_prompt, temperature=0.0)
    out_temp07 = call_llm(system_prompt, user_prompt, temperature=0.7)
    results.append({
        "Input": inp,
        "Output_temp0": out_temp0,
        "Output_temp07": out_temp07
    })

# Print results
for r in results:
    print("Input:", r["Input"])
    print("Output at temp=0:", r["Output_temp0"])
    print("Output at temp=0.7:", r["Output_temp07"])
    print("-"*80)

#3. Structured output handling (all tracks):
# Load the best-performing model
# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level (to CapstoneProject), then into Part1\Output
output_dir_path = os.path.join(script_dir, "..", "Part3", "Output")
# Normalize the path
output_dir_path = os.path.normpath(output_dir_path)
output_file_path = os.path.join(output_dir_path, "best_model.pkl")
# Check folder and file existence
if not os.path.exists(output_dir_path):
    raise FileNotFoundError(f"Required folder not found: {output_dir_path}")
if not os.path.isfile(output_file_path):
    raise FileNotFoundError(f"Required file not found: {output_file_path}")
loaded_model = joblib.load(output_file_path)
#ColumnTransformer with OneHotEncoder
expected_columns = loaded_model.named_steps["simpleimputer"].feature_names_in_
print("Expected feature names:", expected_columns)
# Helper function to expand raw inputs into full encoded rows:
def make_test_row(raw_features: dict, expected_columns: list):
    df = pd.DataFrame([raw_features])
    # Ensure all expected columns exist, fill missing with 0
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

# Three handcrafted feature-vector inputs (replace with realistic feature names/values)
inputs = [
    {"defect_ratio": 0.90, "severity": 2, "waferIndex": 1.2, "dieSize": 0.8, "lotName": 3},
    {"defect_ratio": 0.05, "severity": 0, "waferIndex": 0.7, "dieSize": 1.5, "lotName": 1},
    {"defect_ratio": 0.30, "severity": 1, "waferIndex": 0.9, "dieSize": 1.0, "lotName": 2}
]

# JSON schema for validation
explanation_schema = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

# Fallback dict if validation fails
def fallback_dict():
    return {k: None for k in explanation_schema["required"]}

# Validate function
def validate_explanation(response_text):
    if response_text is None:
        print("LLM call failed: no response text")
        return fallback_dict()
    try:
        parsed = json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return fallback_dict()
    try:
        validate(instance=parsed, schema=explanation_schema)
        return parsed
    except ValidationError as e:
        print("Schema validation error:", e)
        return fallback_dict()

# Zero-shot system prompt
system_prompt = """
You are an assistant that explains machine learning model predictions.
Given feature values, a predicted class, and a predicted probability, output only valid JSON.
The JSON must contain exactly these fields:
{
  "prediction_label": "string",
  "confidence_level": "low|medium|high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}
Confidence rubric:
- probability <0.6 → low
- 0.6–0.8 → medium
- >0.8 → high
Do not include any text outside of the JSON object.
"""

# Loop through inputs
for features in inputs:
    encoded = make_test_row(features, expected_columns)
    pred_class = loaded_model.predict(encoded)[0]
    pred_proba = loaded_model.predict_proba(encoded)[0][pred_class]

    # Construct user prompt
    user_prompt = json.dumps({
        "features": features,
        "prediction_label": str(pred_class),
        "prediction_probability": float(pred_proba)
    })

    # Call LLM (temperature=0 for reproducibility)
    explanation = call_llm(system_prompt, user_prompt, temperature=0.0)

    # Validate JSON
    validated_output = validate_explanation(explanation)

    # Print results
    print("Features:", features)
    print("Predicted Class:", pred_class)
    print("Predicted Probability:", pred_proba)
    print("LLM Explanation:", explanation)
    print("Validation Outcome:", validated_output)
    print("-"*80)

#4. Guardrails
def has_pii(text: str) -> bool:
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
# Example user prompt construction
user_prompt = json.dumps({
    "features": features,
    "prediction_label": str(pred_class),
    "prediction_probability": float(pred_proba)
})

# Guardrail check
if has_pii(user_prompt):
    print("Input blocked: PII detected.")
    explanation = None
else:
    explanation = call_llm(system_prompt, user_prompt, temperature=0.0)

# Test input with email (should be blocked)
test_input_email = "This record belongs to kalaivanicd@masai.com"
if has_pii(test_input_email):
    print("Input blocked: PII detected.")
else:
    explanation = call_llm(system_prompt, test_input_email, temperature=0.0)
    print("LLM Explanation:", explanation)

# Test input without PII (should proceed)
test_input_clean = "Features: defect_ratio=0.3, waferIndex=0.9"
if has_pii(test_input_clean):
    print("Input blocked: PII detected.")
else:
    explanation = call_llm(system_prompt, test_input_clean, temperature=0.0)
    print("LLM Explanation:", explanation)


#5. Demonstrate the feature end-to-end:
import os
import requests
import json
import joblib
import pandas as pd
from jsonschema import validate, ValidationError
import re
# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level (to CapstoneProject), then into Part1\Output
output_dir_path = os.path.join(script_dir, "..", "Part3", "Output")
# Normalize the path
output_dir_path = os.path.normpath(output_dir_path)
output_file_path = os.path.join(output_dir_path, "best_model.pkl")
# Check folder and file existence
if not os.path.exists(output_dir_path):
    raise FileNotFoundError(f"Required folder not found: {output_dir_path}")
if not os.path.isfile(output_file_path):
    raise FileNotFoundError(f"Required file not found: {output_file_path}")
loaded_model = joblib.load(output_file_path)
expected_columns = loaded_model.named_steps["simpleimputer"].feature_names_in_

# Guardrail: PII detection
def has_pii(text: str) -> bool:
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

# Encode helper
def encode_record(raw_features: dict, expected_columns: list):
    df = pd.DataFrame([raw_features])
    return df.reindex(columns=expected_columns, fill_value=0)

# Schema
explanation_schema = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low","medium","high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label","confidence_level","top_reason","second_reason","next_step"]
}

def fallback_dict():
    return {k: None for k in explanation_schema["required"]}

def clean_llm_output(text: str) -> str:
    # Remove Markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned

def map_confidence(prob: float) -> str:
    if prob >= 0.8:
        return "high"
    elif prob >= 0.5:
        return "medium"
    else:
        return "low"

# --- Flatten nested structures into required schema ---
def enforce_schema(parsed: dict) -> dict:
    if "explanation" in parsed:
        exp = parsed["explanation"]
        prob = exp.get("prediction_probability") or exp.get("confidence_level")
        return {
            "prediction_label": exp.get("prediction_label") or exp.get("prediction", {}).get("label"),
            "confidence_level": map_confidence(float(prob)) if prob is not None else "medium",
            "top_reason": exp.get("interpretation"),
            "second_reason": exp.get("overall_assessment") or exp.get("overall_analysis"),
            "next_step": "Review manufacturing process"
        }
    else:
        # If already flat
        prob = parsed.get("confidence_level")
        if isinstance(prob, (int,float)):
            parsed["confidence_level"] = map_confidence(prob)
        return parsed

def validate_explanation(response_text):
    if response_text is None:
        print("LLM call failed: no response text")
        return fallback_dict()
    try:
        cleaned = clean_llm_output(response_text)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        print("JSON decode error:", e)
        return fallback_dict()
    try:
        flattened = enforce_schema(parsed)
        validate(instance=flattened, schema=explanation_schema)
        return flattened
    except ValidationError as e:
        print("Schema validation error:", e)
        return fallback_dict()
    
# Helper function to expand raw inputs into full encoded rows:
def make_test_row(raw_features: dict, expected_columns: list):
    df = pd.DataFrame([raw_features])
    # Ensure all expected columns exist, fill missing with 0
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

#Set up the LLM API connection:
LLM_API_KEY = os.environ["LLM_API_KEY"]
if not LLM_API_KEY:
    raise ValueError("Missing API key. Set LLM_API_KEY environment variable.") 

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",  # passkey retrieved securely
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Error:", response.status_code)
        return None
    return response.json()["choices"][0]["message"]["content"]


system_prompt = """You are an assistant that explains ML predictions in valid JSON.
Always output with keys: prediction_label, confidence_level, top_reason, second_reason, next_step.
"""

# Three test inputs
inputs = [
    {"defect_ratio":0.5,"severity":2,"waferIndex":1.2,"dieSize":0.8,"lotName":3},   # clean
    {"defect_ratio":0.05,"severity":0,"waferIndex":0.7,"dieSize":1.5,"lotName":1}, # clean
    {"defect_ratio":0.3,"severity":1,"waferIndex":0.9,"dieSize":1.0,"lotName":2,"note":"Contact kalaivanicd@masai.com"} # contains PII
]

results = []
for features in inputs:
    encoded = make_test_row(features, expected_columns)
    pred_class = loaded_model.predict(encoded)[0]
    pred_proba = loaded_model.predict_proba(encoded)[0][pred_class]

    user_prompt = json.dumps({
        "features": features,
        "prediction_label": str(pred_class),
        "prediction_probability": float(pred_proba)
    })

    if has_pii(user_prompt):
        explanation = "Input blocked: PII detected."
        validated_output = fallback_dict()
        guardrail_result = "Block"
    else:
        explanation = call_llm(system_prompt, user_prompt, temperature=0.0)
        validated_output = validate_explanation(explanation)
        guardrail_result = "Pass"

    print("Input:", features)
    print("LLM Output:", explanation)
    print("Validation Outcome:", validated_output)
    print("Guardrail:", guardrail_result)
    print("-"*80)

    results.append((features, explanation, validated_output, guardrail_result))