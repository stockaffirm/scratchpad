# pip install langchain langchain_aws langgraph boto3
"""
Advanced Project Code Analyzer using LangGraph on AWS Bedrock.

This script defines a graph where one model analyzes a file, a second model
critiques the analysis, and the first model self-corrects based on the critique.
This process is repeated for each file before a final synthesis report is generated.

Dependencies:
- langchain, langchain_aws, langgraph, boto3
  Install using: pip install langchain langchain_aws langgraph boto3

Setup:
1.  Install and configure the AWS CLI with your credentials (`aws configure`).
2.  Ensure you have access to the models in the Bedrock console.

Usage:
   python analyze_project_langgraph.py /path/to/your/project/directory
"""

import os
import json
import logging
import argparse
from typing import TypedDict, List, Annotated
import operator

from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, END

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Selection from your provided list ---
MODEL_1_ANALYST_ARN = "arn:aws:bedrock:us-east-1:982945613320:application-inference-profile/wwshs11qcgkg3"  # Claude 3.5 Sonnet
MODEL_2_REVIEWER_ARN = "arn:aws:bedrock:us-east-1:982945613320:application-inference-profile/ps5itp0ecmvd"  # Claude 3 Haiku

# --- LangChain Model Initialization ---
# We initialize the models once for reuse.
analyst_model = ChatBedrock(
    model_id=MODEL_1_ANALYST_ARN,
    model_kwargs={"max_tokens": 4096, "temperature": 0.2, "anthropic_version": "bedrock-2023-05-31"}
)

reviewer_model = ChatBedrock(
    model_id=MODEL_2_REVIEWER_ARN,
    model_kwargs={"max_tokens": 2048, "temperature": 0.4, "anthropic_version": "bedrock-2023-05-31"}
)


# --- LangGraph State Definition ---
class AnalysisGraphState(TypedDict):
    """Defines the state that flows through the graph for a single file."""
    file_path: str
    file_content: str
    initial_analysis: str
    critique: str
    # The `operator.add` below means that when this key is updated, the new value is appended to the list.
    all_final_analyses: Annotated[List[str], operator.add]
    final_report: str


# --- LangGraph Nodes ---

def analyze_node(state: AnalysisGraphState):
    """Node 1: Performs the initial analysis of a file."""
    logger.info(f"Node: Analyzing file -> {state['file_path']}")
    prompt = f"""You are a senior software engineer. Analyze the following code file. Focus on its primary purpose, key functions/classes, inputs, outputs, and dependencies.

    CODE FILE: `{state['file_path']}`
    ```
    {state['file_content']}
    ```"""
    response = analyst_model.invoke(prompt)
    return {"initial_analysis": response.content}


def critique_node(state: AnalysisGraphState):
    """Node 2: Critiques the initial analysis."""
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
    """Node 3: Creates a final, corrected analysis based on the critique."""
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
    # We return a list so it can be appended to the state's list
    return {"all_final_analyses": [final_analysis_text]}


def synthesize_node(state: AnalysisGraphState):
    """Final Node: Creates the holistic report from all individual analyses."""
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


# --- Helper Function ---
def read_and_filter_files(directory: str) -> list:
    """Returns a list of valid file paths to analyze."""
    # (Same as previous script, kept for brevity)
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


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Perform an advanced, LangGraph-based analysis of a project directory.")
    parser.add_argument("project_directory", type=str, help="The path to the project directory to analyze.")
    args = parser.parse_args()

    if not os.path.isdir(args.project_directory):
        logger.error(f"Error: Directory not found at '{args.project_directory}'")
        return

    # 1. Define the graph structure
    workflow = StateGraph(AnalysisGraphState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("correct", correct_node)
    workflow.add_node("synthesize", synthesize_node)

    # 2. Define the graph edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "critique")
    workflow.add_edge("critique", "correct")

    # This is a placeholder for the loop logic. We will call the graph for each file.
    # After correction, we end this part of the graph.
    workflow.add_edge("correct", END)

    # Compile the graph for file-level analysis
    file_analysis_app = workflow.compile()

    # 3. Execute the workflow
    file_paths = read_and_filter_files(args.project_directory)

    # This will hold the state as we iterate
    final_state = {"all_final_analyses": []}

    for i, file_path in enumerate(file_paths):
        logger.info(f"--- Processing file {i + 1}/{len(file_paths)}: {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if not content.strip():
                logger.warning(f"Skipping empty file: {file_path}")
                continue

            # Invoke the graph for one file
            result = file_analysis_app.invoke({
                "file_path": file_path,
                "file_content": content,
                "all_final_analyses": []  # Start fresh for each invocation's return
            })
            # Accumulate the results
            final_state["all_final_analyses"].extend(result["all_final_analyses"])

        except Exception as e:
            logger.error(f"Failed to process file {file_path}. Error: {e}")

    # 4. Run the final synthesis step
    if not final_state["all_final_analyses"]:
        logger.error("No files were successfully analyzed. Exiting.")
        return

    logger.info("--- All files processed. Starting final synthesis. ---")
    final_report_state = synthesize_node(final_state)

    # 5. Print the final report
    print("\n\n" + "=" * 80)
    print("      HOLISTIC PROJECT ANALYSIS AND DATA FLOW REPORT (via LangGraph)")
    print("=" * 80 + "\n")
    print(final_report_state['final_report'])


if __name__ == "__main__":
    main()