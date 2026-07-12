# ------WM811K Wafer Map Dataset: From Data Cleaning to Explainable AI------ #

Choosen the WM‑811K semiconductor wafer map dataset for the Applied AI & ML Essentials — Capstone Project, because it is one of the largest publicly available, real‑world datasets for defect pattern recognition in semiconductor manufacturing.

# PreRequiste 
Make sure the tuned Random Forest model best_model.pkl is available from Part 3 inside the Part3/Output (or) Run Part1 python script and continue with Part3

# Installation 
Install the required library to run the python script in Part4 folder
```bash 
    pip install pandas requests json jconscheme os joblib
```
# OS
Windows OS
# IDE
Visual Studio Code
# OpenRouter URL
"https://openrouter.ai/api/v1/chat/completions"
# model
"openai/gpt-4o-mini"
# Get, Store and Access OpenRouter API key
1. Obtain Your API Key
First, generate your OpenRouter API key from your account dashboard.
Log in to OpenRouter
Navigate to API Keys section
Copy the generated key securely
The key can be accessed in twp way either follow step 2 and step 3 (or) step 4 and step 5
2. Set the environment variable: (LLM.py in Part 4 follow this approach)
For Windows OS
Open the VS Code terminal and run:
```bash
setx LLM_API_KEY "your_api_key_here" 
```
Now restart VS Code so the new variable is available.
or Store your API key in an system environment variable(LLM_API_KEY) before running
3. Access the stored key with
```python
import os
LLM_API_KEY = os.environ["LLM_API_KEY"]
```
4. Create a .env File
  Best Practice
  Store secrets in a .env file excluded from version control.
Create a file named .env 
Add the line: LLM_API_KEY=your_api_key_here
Ensure .env is listed in .gitignore
Make sure dotenv is installed
``` pip install python-dotenv```
5. Load Environment Variable in Python
Access the stored key in your code using os.environ.
```python
import os
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()
LLM_API_KEY = os.environ["LLM_API_KEY"]
#Alternatively, use os.environ.get('LLM_API_KEY') to avoid KeyError
```
# Run the python script
LLM.py for Part 4

# Part 4 — LLM-Powered Feature: Structured Extraction, Tabular Batch Scoring, or Model Prediction Explanation
# LLM-powered feature that demonstrates prompt engineering, structured output handling, and basic safety guardrails on WM811K cleaned Dataset
# Scenario
The client wants an LLM-powered feature layered on top of the data work already done. You will build one LLM-powered feature that demonstrates prompt engineering, structured output handling, and basic safety guardrails. You will use a public LLM API (any provider that accepts HTTP POST requests with a JSON body containing model and messages fields, such as OpenRouter) accessed via the Python requests library.

# PRE-TASK
# Choose exactly one of the following three feature tracks and state your choice
# (C) Model Prediction Explanation Pipeline: 
Load the best-performing model from Part 3 using joblib.load('best_model.pkl'). For three hand-crafted feature-vector inputs, call .predict() and .predict_proba(). For each input, pass the feature values, predicted class, and predicted probability to the LLM API as a structured user prompt and request a JSON explanation with at least 5 required scalar fields (e.g., {"prediction_label": "string", "confidence_level": "low|medium|high", "top_reason": "string", "second_reason": "string", "next_step": "string"}). Validate each JSON response against a schema and apply the PII guardrail before each LLM call.
# Why
The client already has a high‑performing Random Forest model that predicts wafer outcomes. What they lack is interpretability — stakeholders want to understand why the model made a decision. Track C directly addresses this by layering an LLM explanation pipeline on top of the existing ML work.


# 1. Set up the LLM API connection:
# Walkthrough
1. Importing Libraries
To access the key stored in the environment variable and used to send HTTP requests to the LLM API endpoint
2. Loading the API key
retrieves the API key from your environment variables.
The print statement confirms that the key was found.
If the key is missing, the script raises an error instead of failing silently.
3. Defining the reusable function
Function parameters:
system_prompt: sets the role or behavior of the assistant.
user_prompt: the actual query or instruction.
temperature: controls randomness (0 = deterministic, higher = more creative).
max_tokens: limits the length of the response.
url: the API endpoint for chat completions.
headers: includes your API key for authentication and specifies JSON format.
4. Constructing the payload
model: specifies which LLM to use.
messages: a list of dictionaries representing the conversation.
The system message defines the assistant’s role.
The user message is the actual input.
temperature and max_tokens are passed directly into the API.
5. Sending the request
requests.post() sends the payload to the API.
If the response code isn’t 200 (success), it prints the error and returns None.
(This prevents your script from crashing unexpectedly)
6. Parse the response
Converts the response into JSON.
Extracts the assistant’s reply from the nested structure:
choices[0] → first completion choice.
message["content"] → the actual text output
7. Demonstration
Defines a simple test prompt.
Calls the function.
Prints the output
LLM output: hello

# Inference
Security: API key is stored in environment variables, not hardcoded.
Reusability: call_llm() can be used across multiple scripts.
Error handling: Checks for missing keys and failed requests.
Clarity: Clean separation of setup, payload, request, and response parsing.

# 2. Prompt design:
# Walkthrough
1. Zero-shot system prompt for Track C
This defines the LLM’s role: structured explainer.
It enforces valid JSON only, with a fixed schema.
The rubric ensures probabilities are mapped consistently to confidence levels.
Stating “Do not include any text outside of the JSON object,” prevent stray text that would break downstream parsing.
2. Test Inputs
Three synthetic wafer feature vectors with predicted labels and probabilities.
These serve as inputs for A/B testing at different temperatures.
3. Running Temperature Comparison
Each input is converted to JSON and passed as the user prompt.
Two calls are made: one with temperature=0.0 (deterministic) and one with temperature=0.7 (variable).
Results are stored for later comparison.
4. Print the Reslults
Displays each input alongside both outputs.
The separator line makes the comparison easy to read.
Input: {'features': {'defect_ratio': 0.5, 'severity': 2}, 'prediction_label': '1', 'prediction_probability': 0.87}
Output at temp=0: {
  "prediction_label": "1",
  "confidence_level": "high",
  "top_reason": "High defect ratio indicates a significant issue.",
  "second_reason": "Severity level suggests the defect is critical.",
  "next_step": "Investigate the defect further and implement corrective actions."
}
Output at temp=0.7: {
  "prediction_label": "1",
  "confidence_level": "high",
  "top_reason": "High defect ratio indicates a significant issue.",
  "second_reason": "Severity level of 2 suggests a moderate impact.",
  "next_step": "Investigate the defects and take corrective actions."
}
--------------------------------------------------------------------------------
Input: {'features': {'defect_ratio': 0.05, 'severity': 0}, 'prediction_label': '0', 'prediction_probability': 0.42}
Output at temp=0: {
  "prediction_label": "0",
  "confidence_level": "low",
  "top_reason": "Low defect ratio indicates minimal issues.",
  "second_reason": "Severity is zero, suggesting no critical problems.",
  "next_step": "Monitor for any changes in defect ratio."
}
Output at temp=0.7: {
  "prediction_label": "0",
  "confidence_level": "low",
  "top_reason": "The defect ratio is low, indicating minimal issues.",
  "second_reason": "The severity level is zero, suggesting no critical defects.",
  "next_step": "Monitor the situation for any changes in defect ratio."
}
--------------------------------------------------------------------------------
Input: {'features': {'defect_ratio': 0.3, 'severity': 1}, 'prediction_label': '0', 'prediction_probability': 0.65}
Output at temp=0: {
  "prediction_label": "0",
  "confidence_level": "medium",
  "top_reason": "Defect ratio is relatively low.",
  "second_reason": "Severity level is minimal.",
  "next_step": "Monitor the situation for any changes."
}
Output at temp=0.7: {
  "prediction_label": "0",
  "confidence_level": "medium",
  "top_reason": "Defect ratio is below the threshold for a positive class.",
  "second_reason": "Severity level is low, indicating minor issues.",
  "next_step": "Monitor the situation and reassess if defect ratio increases."
}
--------------------------------------------------------------------------------

# Inference
System Prompt: Defines the LLM’s role as a structured explainer of ML predictions.
Zero‑shot design: No examples are provided; the schema and rubric are sufficient for deterministic behavior.
Temperature: Always set to 0.0 in your call_llm function for reproducible, predictable outputs — the model consistently selects the highest‑probability next token, which is critical for schema compliance.

# write out the exact system prompt and the user prompt template (with placeholders shown).
# System prompt:
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

# User prompt template:
Feature values: {feature_vector}
Predicted class: {predicted_class}
Predicted probability: {predicted_probability}

Generate a JSON explanation.


# Explain why you chose temperature=0 for this task (consistent with the rule that low temperature near 0 results in deterministic, predictable outputs suitable for structured data tasks).
For this task, we set temperature=0 because:
Deterministic outputs: At temperature=0, the model always selects the highest-probability next token, ensuring consistent JSON structure and wording across runs.
Schema compliance: Structured data tasks require predictable formatting. Variability could break schema validation or introduce inconsistent phrasing.
Reproducibility: Explanations must be exam-ready and repeatable. Using temperature=0 guarantees that the same input produces the same output every time.

# Table with columns: Input, Output at temp=0, Output at temp=0.7, Key difference
| Input (features + prediction) | Output at temp=0 | Output at temp=0.7 | Key Difference |
| --- | --- | --- | --- |
| ``{'features': ``{'defect_ratio': ``0.5, ``'severity': ``2}, ``'prediction_label': ``'1', ``'prediction_probability': ``0.87}`` | ``{ ``"prediction_label": ``"1", ``"confidence_level": ``"high", ``"top_reason": ``"High ``defect ``ratio ``indicates ``a ``significant ``issue.", ``"second_reason": ``"Severity ``level ``suggests ``the ``defect ``is ``critical.", ``"next_step": ``"Investigate ``the ``defect ``further ``and ``implement ``corrective ``actions." ``}`` | ``{ ``"prediction_label": ``"1", ``"confidence_level": ``"high", ``"top_reason": ``"High ``defect ``ratio ``indicates ``a ``significant ``issue.", ``"second_reason": ``"Severity ``level ``of ``2 ``suggests ``a ``moderate ``impact.", ``"next_step": ``"Investigate ``the ``defects ``and ``take ``corrective ``actions." ``}`` | Temp=0 output is consistent; temp=0.7 varies severity interpretation and next step phrasing. |
| ``{'features': ``{'defect_ratio': ``0.05, ``'severity': ``0}, ``'prediction_label': ``'0', ``'prediction_probability': ``0.42}`` | ``{ ``"prediction_label": ``"0", ``"confidence_level": ``"low", ``"top_reason": ``"Low ``defect ``ratio ``indicates ``minimal ``issues.", ``"second_reason": ``"Severity ``is ``zero, ``suggesting ``no ``critical ``problems.", ``"next_step": ``"Monitor ``for ``any ``changes ``in ``defect ``ratio." ``}`` | ``{ ``"prediction_label": ``"0", ``"confidence_level": ``"low", ``"top_reason": ``"The ``defect ``ratio ``is ``low, ``indicating ``minimal ``issues.", ``"second_reason": ``"The ``severity ``level ``is ``zero, ``suggesting ``no ``critical ``defects.", ``"next_step": ``"Monitor ``the ``situation ``for ``any ``changes ``in ``defect ``ratio." ``}`` | Temp=0 output is deterministic; temp=0.7 introduces synonyms and slightly different phrasing. |
| ``{'features': ``{'defect_ratio': ``0.3, ``'severity': ``1}, ``'prediction_label': ``'0', ``'prediction_probability': ``0.65}`` | ``{ ``"prediction_label": ``"0", ``"confidence_level": ``"medium", ``"top_reason": ``"Defect ``ratio ``is ``relatively ``low.", ``"second_reason": ``"Severity ``level ``is ``minimal.", ``"next_step": ``"Monitor ``the ``situation ``for ``any ``changes." ``}`` | ``{ ``"prediction_label": ``"0", ``"confidence_level": ``"medium", ``"top_reason": ``"Defect ``ratio ``is ``below ``the ``threshold ``for ``a ``positive ``class.", ``"second_reason": ``"Severity ``level ``is ``low, ``indicating ``minor ``issues.", ``"next_step": ``"Monitor ``the ``situation ``and ``reassess ``if ``defect ``ratio ``increases." ``}`` | Temp=0 output is stable; temp=0.7 varies phrasing of reasons and adds nuance to next step. |

# why temperature=0 produces more deterministic outputs (the model always picks the highest-probability next token) and why temperature=0.7 introduces variability (it samples from a broader distribution).
Temperature = 0: Produces deterministic outputs. The model always picks the highest‑probability next token, ensuring consistent JSON structure and wording across runs. This is ideal for structured data tasks where schema compliance and reproducibility are critical.
Temperature = 0.7: Introduces variability by sampling from a broader probability distribution. This results in different phrasings, synonyms, or reasoning steps. While useful for creative tasks, it can reduce consistency and reliability in structured JSON generation.

# 3. Structured output handling (Track C):
# Walkthrough
1. Load the Best Model
Uses joblib.load to deserialize the trained pipeline stored in best_model.pkl.
The pipeline likely includes preprocessing steps (like SimpleImputer, OneHotEncoder) and a classifier.
2. Extract Expected Columns
Accesses the SimpleImputer step inside the pipeline (named_steps["simpleimputer"]).
Retrieves the feature names the imputer expects (feature_names_in_).
This ensures that when you build test rows, they align with the model’s training schema.
3. Helper Function for Encoding Inputs
Converts a raw dictionary of features into a Pandas DataFrame.
Reindexes the DataFrame to match the exact expected column order.
Any missing columns are filled with 0 (safe default).
This guarantees compatibility with the trained pipeline.
4. Handcrafted Test Inputs
Three synthetic feature vectors simulating wafer defect data.
Each dict contains numeric values for features like defect_ratio, severity, waferIndex, etc.
These will be passed through the pipeline for prediction.
5. JSON Schema for Validation
Defines the expected structure of the LLM’s explanation output.
Ensures the JSON has exactly five fields with correct types and allowed values.
Acts as a contract between the LLM and your pipeline.
6. Fallback Dictionary
Provides a safe default if parsing or validation fails.
Returns all required fields with None values.
Prevents downstream code from breaking due to missing keys.
7. Validation Function
Handles three possible failure points:
No response → fallback.
Invalid JSON → fallback.
Schema mismatch → fallback.
If all checks pass, returns the parsed JSON explanation.
8. Zero‑Shot System Prompt
Instructs the LLM to produce only valid JSON with the required fields.
Includes a confidence rubric based on probability thresholds:
<0.6 → low
0.6–0.8 → medium
>0.8 → high
Ensures reproducibility and structured output.
9. Main Loop
Encode input → ensures alignment with training schema.
Predict class → model outputs 0/1 (or other labels).
Predict probability → extracts probability for the predicted class.
Construct prompt → passes features, class, and probability to the LLM.
Call LLM → generates explanation JSON (temperature=0 ensures deterministic output).
Validate → ensures JSON matches schema, fallback if not.
Print results → shows raw features, predictions, explanation, and validation outcome.

Expected feature names: ['waferIndex' 'defect_count' 'normal_count' 'defect_ratio'
 "trianTestLabel_[['Training']]" "failureType_[['Donut']]"
 "failureType_[['Edge-Loc']]" "failureType_[['Edge-Ring']]"
 "failureType_[['Loc']]" "failureType_[['Near-full']]"
 "failureType_[['Random']]" "failureType_[['Scratch']]"
 "failureType_[['none']]"]
Features: {'defect_ratio': 0.9, 'severity': 2, 'waferIndex': 1.2, 'dieSize': 0.8, 'lotName': 3}
Predicted Class: 0
Predicted Probability: 0.915
LLM Explanation: {
  "prediction_label": "0",
  "confidence_level": "high",
  "top_reason": "High defect ratio indicates a strong likelihood of class 0.",
  "second_reason": "Severity level suggests significant issues that align with class 0 characteristics.",
  "next_step": "Monitor the production process closely and investigate the causes of defects."
}
Validation Outcome: {'prediction_label': '0', 'confidence_level': 'high', 'top_reason': 'High defect ratio indicates a strong likelihood of class 0.', 'second_reason': 'Severity level suggests significant issues that align with class 0 characteristics.', 'next_step': 'Monitor the production process closely and investigate the causes of defects.'}
--------------------------------------------------------------------------------
Features: {'defect_ratio': 0.05, 'severity': 0, 'waferIndex': 0.7, 'dieSize': 1.5, 'lotName': 1}
Predicted Class: 0
Predicted Probability: 0.845
LLM Explanation: {
  "prediction_label": "0",
  "confidence_level": "high",
  "top_reason": "The defect ratio is very low, indicating a high likelihood of quality.",
  "second_reason": "The severity is zero, suggesting no critical issues detected.",
  "next_step": "Proceed with production as the quality is expected to be high."
}
Validation Outcome: {'prediction_label': '0', 'confidence_level': 'high', 'top_reason': 'The defect ratio is very low, indicating a high likelihood of quality.', 'second_reason': 'The severity is zero, suggesting no critical issues detected.', 'next_step': 'Proceed with production as the quality is expected to be high.'}
--------------------------------------------------------------------------------
Features: {'defect_ratio': 0.3, 'severity': 1, 'waferIndex': 0.9, 'dieSize': 1.0, 'lotName': 2}
Predicted Class: 0
Predicted Probability: 0.94
LLM Explanation: {
  "prediction_label": "0",
  "confidence_level": "high",
  "top_reason": "High wafer index indicates a strong likelihood of class 0.",
  "second_reason": "Low defect ratio suggests fewer issues in the lot.",
  "next_step": "Proceed with production as the model indicates a high confidence in this prediction."
}
Validation Outcome: {'prediction_label': '0', 'confidence_level': 'high', 'top_reason': 'High wafer index indicates a strong likelihood of class 0.', 'second_reason': 'Low defect ratio suggests fewer issues in the lot.', 'next_step': 'Proceed with production as the model indicates a high confidence in this prediction.'}
--------------------------------------------------------------------------------

# Inference
Handcrafted wafer defect vectors aligned with the model’s expected schema, direct outputs from the trained pipeline (predict + predict_proba), generated by the LLM with temperature=0 for deterministic, reproducible output. All three outputs passed schema validation, confirming robustness of the structured pipeline.

# Track C Structured Output Demonstratio
| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
| --- | --- | --- | --- | --- |
| ``{"defect_ratio":0.9,"severity":2,"waferIndex":1.2,"dieSize":0.8,"lotName":3}`` | 0 | 0.915 | ``{ ``"prediction_label":"0","confidence_level":"high","top_reason":"High ``defect ``ratio ``indicates ``a ``strong ``likelihood ``of ``class ``0.","second_reason":"Severity ``level ``suggests ``significant ``issues ``that ``align ``with ``class ``0 ``characteristics.","next_step":"Monitor ``the ``production ``process ``closely ``and ``investigate ``the ``causes ``of ``defects." ``}`` | Passed |
| ``{"defect_ratio":0.05,"severity":0,"waferIndex":0.7,"dieSize":1.5,"lotName":1}`` | 0 | 0.845 | ``{ ``"prediction_label":"0","confidence_level":"high","top_reason":"The ``defect ``ratio ``is ``very ``low, ``indicating ``a ``high ``likelihood ``of ``quality.","second_reason":"The ``severity ``is ``zero, ``suggesting ``no ``critical ``issues ``detected.","next_step":"Proceed ``with ``production ``as ``the ``quality ``is ``expected ``to ``be ``high." ``}`` | Passed |
| ``{"defect_ratio":0.3,"severity":1,"waferIndex":0.9,"dieSize":1.0,"lotName":2}`` | 0 | 0.94 | ``{ ``"prediction_label":"0","confidence_level":"high","top_reason":"High ``wafer ``index ``indicates ``a ``strong ``likelihood ``of ``class ``0.","second_reason":"Low ``defect ``ratio ``suggests ``fewer ``issues ``in ``the ``lot.","next_step":"Proceed ``with ``production ``as ``the ``model ``indicates ``a ``high ``confidence ``in ``this ``prediction." ``}`` | Passed |

# 4. Guardrails:
# Walkthrough 
1.Guardrail Function
Purpose: Detects personally identifiable information (PII) in text before sending it to the LLM.
Regex patterns:
email_pattern: Matches standard email addresses like kalaivanicd@masai.com.
phone_pattern: Matches either:
1. consecutive digits (1234567890), or
US‑style formatted numbers (123-456-7890, 123.456.7890, 123 456 7890).
Return value: True if either pattern is found, otherwise False.
2. Guardrail Integration
Before calling the LLM, the guardrail checks the constructed user_prompt.
If PII is detected → block the request (explanation = None).
If clean → proceed with the LLM call.
3. Test Case 1: Input with Email (Blocked)
Regex match: Finds kalaivanicd@example.com.
Result: "Input blocked: PII detected."
LLM call: Skipped.
4. Test Case 2: Input without PII (Allowed)
Regex match: No email or phone number detected.
Result: Guardrail passes.
LLM call: Executes, explanation is printed.
LLM Explanation: {
  "prediction_label": "defective",
  "confidence_level": "medium",
  "top_reason": "defect_ratio is above the threshold for defects",
  "second_reason": "waferIndex indicates a high likelihood of defects",
  "next_step": "review the manufacturing process for quality control"
}

# Inference
Ensure sensitive data (emails, phone numbers) never reaches the LLM. Blocked inputs are logged transparently.

# Test Case 1: Input with Email (Blocked)
```python
test_input_email = "This record belongs to kalaivanicd@example.com"
safe_call_llm("system_prompt", test_input_email)
```
Output:
```
Input blocked: PII detected.
```

# Test Case 2: Input without PII (Allowed)
```python
test_input_clean = "Features: defect_ratio=0.3, waferIndex=0.9"
safe_call_llm("system_prompt", test_input_clean)
```
Output:
```json
LLM Explanation: {
  "prediction_label": "defective",
  "confidence_level": "medium",
  "top_reason": "defect_ratio is above the threshold for defects",
  "second_reason": "waferIndex indicates a high likelihood of defects",
  "next_step": "review the manufacturing process for quality control"
}
```

# 5. Demonstrate the feature end-to-end:
# Walkthrough
1. Model Loading
Loads the trained pipeline (best_model.pkl).
Extracts the expected feature names from the preprocessing step (simpleimputer), ensuring that any input records are aligned with the model’s schema.
2. Guardrail: PII Detection
Uses regex to detect emails and phone numbers.
If PII is found, the pipeline blocks the LLM call.
3. Input Encoding
Converts raw feature dictionaries into a DataFrame.
Ensures all expected columns are present, filling missing ones with 0.
4. Schema Definition
Defines the expected JSON structure for LLM explanations.
Guarantees consistency across outputs.
5. Utility Functions in the end‑to‑end feature
fallback_dict(): Returns a null‑filled dict if validation fails.
clean_llm_output(): Strips Markdown fences from LLM responses.
map_confidence(): Maps numeric probabilities to "low", "medium", "high".
enforce_schema(): Flattens nested structures into the required schema.
validate_explanation(): Cleans, parses, flattens, and validates the LLM output against the schema.
make_test_row(): Helper function to expand raw inputs into full encoded rows
call_llm(): Set up the callable and reusable LLM function
6. System Prompt
Instructs the LLM to always return structured JSON explanations.
7. Test Inputs
Two clean feature vectors.
One vector with an email in the note field (to trigger guardrail).
8. Pipeline Execution
Prediction: Runs the model to get class + probability.
Prompt construction: Builds JSON for the LLM.
Guardrail check: Blocks if PII is detected.
LLM call + validation: If clean, explanation is generated and validated.
Output: Prints input, raw LLM output, validation outcome, and guardrail status.

Input: {'defect_ratio': 0.5, 'severity': 2, 'waferIndex': 1.2, 'dieSize': 0.8, 'lotName': 3}
LLM Output: {
  "prediction_label": "0",
  "confidence_level": 0.92,
  "top_reason": "The defect ratio is relatively high at 0.5, indicating a significant likelihood of issues.",
  "second_reason": "The severity level of 2 suggests that the defects present are moderate, contributing to the prediction.",
  "next_step": "Monitor the production closely and consider implementing quality control measures to address the defect ratio."
}
Validation Outcome: {'prediction_label': '0', 'confidence_level': 'high', 'top_reason': 'The defect ratio is relatively high at 0.5, indicating a significant likelihood of issues.', 'second_reason': 'The severity level of 2 suggests that the defects present are moderate, contributing to the prediction.', 'next_step': 'Monitor the production closely and consider implementing quality control measures to address the defect ratio.'}
Guardrail: Pass
--------------------------------------------------------------------------------
Input: {'defect_ratio': 0.05, 'severity': 0, 'waferIndex': 0.7, 'dieSize': 1.5, 'lotName': 1}
LLM Output: {
  "prediction_label": "0",
  "confidence_level": 0.845,
  "top_reason": "The defect ratio is low at 0.05, indicating a high likelihood of quality.",
  "second_reason": "The severity level is 0, suggesting no significant issues detected.",
  "next_step": "Continue monitoring the production process and maintain quality checks."
}
Validation Outcome: {'prediction_label': '0', 'confidence_level': 'high', 'top_reason': 'The defect ratio is low at 0.05, indicating a high likelihood of quality.', 'second_reason': 'The severity level is 0, suggesting no significant issues detected.', 'next_step': 'Continue monitoring the production process and maintain quality checks.'}
Guardrail: Pass
--------------------------------------------------------------------------------
Input: {'defect_ratio': 0.3, 'severity': 1, 'waferIndex': 0.9, 'dieSize': 1.0, 'lotName': 2, 'note': 'Contact kalaivanicd@masai.com'}
LLM Output: Input blocked: PII detected.
Validation Outcome: {'prediction_label': None, 'confidence_level': None, 'top_reason': None, 'second_reason': None, 'next_step': None}
Guardrail: Block
--------------------------------------------------------------------------------

# Inference
Safety was ensured with Guardrail blocking inputs containing PII. Structure Schema validation ensures consistent JSON outputs. Interpretability with LLM explanations highlight key features influencing predictions.

# Track C End‑to‑End Demonstration
| Input | LLM Output | Valid JSON (Pass/Fail) | Guardrail Result |
| --- | --- | --- | --- |
| ``{"defect_ratio":0.5,"severity":2,"waferIndex":1.2,"dieSize":0.8,"lotName":3}`` | ``{ ``"prediction_label":"0","confidence_level":0.92,"top_reason":"The ``defect ``ratio ``is ``relatively ``high ``at ``0.5, ``indicating ``a ``significant ``likelihood ``of ``issues.","second_reason":"The ``severity ``level ``of ``2 ``suggests ``that ``the ``defects ``present ``are ``moderate, ``contributing ``to ``the ``prediction.","next_step":"Monitor ``the ``production ``closely ``and ``consider ``implementing ``quality ``control ``measures ``to ``address ``the ``defect ``ratio." ``}`` | Pass  | Pass  |
| ``{"defect_ratio":0.05,"severity":0,"waferIndex":0.7,"dieSize":1.5,"lotName":1}`` | ``{ ``"prediction_label":"0","confidence_level":0.845,"top_reason":"The ``defect ``ratio ``is ``low ``at ``0.05, ``indicating ``a ``high ``likelihood ``of ``quality.","second_reason":"The ``severity ``level ``is ``0, ``suggesting ``no ``significant ``issues ``detected.","next_step":"Continue ``monitoring ``the ``production ``process ``and ``maintain ``quality ``checks." ``}`` | Pass  | Pass  |
| ``{"defect_ratio":0.3,"severity":1,"waferIndex":0.9,"dieSize":1.0,"lotName":2,"note":"Contact ``kalaivanicd@masai.com"}`` | ``Input ``blocked: ``PII ``detected.`` | Fail (fallback dict returned) | Block  |


#### SUMMARY ####
Part 4 demonstrates a production-ready pipeline where ML predictions are explained by an LLM under strict safety and validation guardrails. Effectively shown how to combine predictive modeling, structured output enforcement, and compliance checks into a single end-to-end feature.
