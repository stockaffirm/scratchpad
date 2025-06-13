"""
Advanced Project Code Analyzer using LangGraph on AWS Bedrock.

This script defines a graph where one model analyzes a file, a second model
critiques the analysis, and the first model self-corrects based on the critique.
This process is repeated for each file.

It includes a code cleaning step to remove comments and docstrings before
sending the code to the models to reduce tokens and improve focus.

Finally, it synthesizes a holistic report and uploads it to an S3 bucket
for ingestion by a Bedrock Knowledge Base.

Dependencies:
- langchain, langchain_aws, langgraph, boto3
  Install using: pip install langchain langchain_aws langgraph boto3

Setup:
1.  Install and configure the AWS CLI with your credentials (`aws configure`).
2.  Ensure you have access to the models in the Bedrock console.
3.  Create an S3 bucket and update the S3_BUCKET_NAME variable below.

Usage:
   python analyze_project_langgraph.py /path/to/your/project/directory
"""

import os
import re
import json
import logging
import argparse
from typing import TypedDict, List, Annotated
import operator
import boto3

from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Selection from your provided list ---
MODEL_1_ANALYST_ARN = "arn:aws:bedrock:us-east-1:982945613320:application-inference-profile/wwshs11qcgkg3" # Claude 3.5 Sonnet
MODEL_2_REVIEWER_ARN = "arn:aws:bedrock:us-east-1:982945613320:application-inference-profile/ps5itp0ecmvd" # Claude 3 Haiku

# --- LangChain Model Initialization ---
analyst_model = ChatBedrock(
    model_id=MODEL_1_ANALYST_ARN,
    model_kwargs={"max_tokens": 4096, "temperature": 0.2, "anthropic_version": "bedrock-2023-05-31"}
)
reviewer_model = ChatBedrock(
    model_id=MODEL_2_REVIEWER_ARN,
    model_kwargs={"max_tokens": 2048, "temperature": 0.4, "anthropic_version": "bedrock-2023-05-31"}
)

# --- Helper Functions ---
def clean_code(code: str) -> str:
    """Removes comments, docstrings, and excessive newlines from a code string."""
    code = re.sub(r"//.*|#.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code, flags=re.MULTILINE)
    code = re.sub(r"'''[\s\S]*?'''", "", code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

def read_and_filter_files(directory: str) -> list:
    """Returns a list of valid file paths to analyze."""
    file_paths = []
    skip_extensions = [
        '.pyc', '.git', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
        '.pdf', '.zip', '.tar', '.gz', '.rar', '.exe', '.dll', '.pyd',
        '.so', '.dylib', '.db', '.sqlite', '.dat', '.json', '.csv', '.tsv',
        '.xlsx', '.xls', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md',
        '.log', '.conf', '.ini', '.yaml', '.yml', '.xml', '.html', '.htm',
    ]
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
        for file in files:
            if not any(file.lower().endswith(ext) for ext in skip_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths

# --- LangGraph State and Nodes ---
class AnalysisGraphState(TypedDict):
    file_path: str
    file_content: str
    initial_analysis: str
    critique: str
    all_final_analyses: Annotated[List[str], operator.add]
    final_report: str

def analyze_node(state: AnalysisGraphState):
    logger.info(f"Node: Analyzing file -> {state['file_path']}")
    prompt = f"""You are a senior software engineer. Analyze the following code file. Focus on its primary purpose, key functions/classes, inputs, outputs, and dependencies.

    CODE FILE: `{state['file_path']}`
    ```
    {state['file_content']}
    ```"""
    response = analyst_model.invoke(prompt)
    return {"initial_analysis": response.content}

def critique_node(state: AnalysisGraphState):
    logger.info(f"Node: Critiquing analysis for -> {state['file_path']}")
    prompt = f"""You are a meticulous code reviewer. Below is a code file and an AI-generated analysis of it. Your task is to critique the analysis. Is it accurate? Is it missing any key logic, side effects, or important details? Be concise and direct.

    CODE FILE: `{state['file_path']}`
    ```
    {state['file_content']}
    ```

    AI ANALYSIS TO REVIEW:
    ---
    {state['initial_analysis']}
    ---"""
    response = reviewer_model.invoke(prompt)
    return {"critique": response.content}

def correct_node(state: AnalysisGraphState):
    logger.info(f"Node: Correcting analysis for -> {state['file_path']}")
    prompt = f"""You are a senior software engineer. You previously provided an analysis of a code file. A reviewer has provided a critique. Now, review the original code, your first analysis, and the critique to generate a final, improved, and comprehensive analysis of the file.

    ORIGINAL CODE FILE: `{state['file_path']}`
    ```
    {state['file_content']}
    ```

    YOUR FIRST ANALYSIS:
    ---
    {state['initial_analysis']}
    ---

    REVIEWER'S CRITIQUE:
    ---
    {state['critique']}
    ---"""
    response = analyst_model.invoke(prompt)
    final_analysis_text = f"--- ANALYSIS FOR FILE: {state['file_path']} ---\n{response.content}\n\n"
    return {"all_final_analyses": [final_analysis_text]}

def synthesize_node(state: AnalysisGraphState):
    logger.info("Node: Synthesizing final holistic report.")
    combined_analyses = "".join(state['all_final_analyses'])
    prompt = f"""
    Based on the following detailed analyses of individual project files, create a holistic, end-to-end report. Please structure your report with the following specific sections:

    1.  **Overall Purpose and End-to-End Flow**
    2.  **Key Components and Interactions**
    3.  **Inferred Input Fields** (with data types)
    4.  **Inferred Output Fields** (with data types)
    5.  **Fields Not Passing to Output**
    6.  **Fields Used in Filters**
    7.  **Specific Values Used in Filters**
    8.  **Data Profiling & Quality Insights**
    9.  **Data Analysis Connections**

    --- INDIVIDUAL FILE ANALYSES ---
    {combined_analyses}
    """
    response = analyst_model.invoke(prompt)
    return {"final_report": response.content}

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Perform an advanced, LangGraph-based analysis of a project directory.")
    parser.add_argument("project_directory", type=str, help="The path to the project directory to analyze.")
    args = parser.parse_args()

    if not os.path.isdir(args.project_directory):
        logger.error(f"Error: Directory not found at '{args.project_directory}'")
        return

    workflow = StateGraph(AnalysisGraphState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("correct", correct_node)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "critique")
    workflow.add_edge("critique", "correct")
    workflow.add_edge("correct", END)
    file_analysis_app = workflow.compile()

    file_paths = read_and_filter_files(args.project_directory)
    final_state = {"all_final_analyses": []}

    for i, file_path in enumerate(file_paths):
        logger.info(f"--- Processing file {i+1}/{len(file_paths)}: {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
            cleaned_content = clean_code(raw_content)
            if not cleaned_content:
                logger.warning(f"Skipping file as it is empty after cleaning: {file_path}")
                continue
            result = file_analysis_app.invoke({
                "file_path": file_path,
                "file_content": cleaned_content,
                "all_final_analyses": []
            })
            final_state["all_final_analyses"].extend(result["all_final_analyses"])
        except Exception as e:
            logger.error(f"Failed to process file {file_path}. Error: {e}")

    if not final_state["all_final_analyses"]:
        logger.error("No files were successfully analyzed. Exiting.")
        return

    logger.info("--- All files processed. Starting final synthesis. ---")
    final_report_state = synthesize_node(final_state)
    final_report = final_report_state['final_report']
    individual_analyses = "".join(final_state['all_final_analyses'])

    S3_BUCKET_NAME = "your-unique-bucket-name-here"
    S3_FILE_NAME = "full_project_analysis_report.txt"

    print("\n\n" + "="*80)
    print(f"      STEP 5: UPLOADING KNOWLEDGE TO S3 BUCKET: {S3_BUCKET_NAME}")
    print("="*80 + "\n")

    full_knowledge_base_text = f"--- HOLISTIC PROJECT REPORT ---\n{final_report}\n\n--- DETAILED FILE-BY-FILE ANALYSIS ---\n{individual_analyses}"

    if S3_BUCKET_NAME == "your-unique-bucket-name-here":
        logger.error("S3_BUCKET_NAME has not been configured. Please update the script.")
        print("\n--- UPLOAD SKIPPED ---\nPlease configure the S3_BUCKET_NAME variable in the script to enable uploads.")
        return

    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_NAME, Body=full_knowledge_base_text.encode('utf-8'))
        logger.info(f"Successfully uploaded report to s3://{S3_BUCKET_NAME}/{S3_FILE_NAME}")
        print("\n--- NEXT STEPS ---\n1. Go to your Knowledge Base in the Amazon Bedrock console.\n2. Select your data source and click the 'Sync' button.\n3. Once syncing is complete, you can use a separate query script to ask questions.")
    except Exception as e:
        logger.error(f"Failed to upload to S3. Error: {e}")
        print("\n--- UPLOAD FAILED ---\nPlease check your AWS credentials and S3 bucket permissions.")

if __name__ == "__main__":
    main()