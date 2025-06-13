"""
Query a Bedrock Knowledge Base containing project code analysis.

This script connects to a pre-configured Bedrock Knowledge Base,
and allows a user to ask questions in a continuous loop.

Usage:
   python query_kb.py
"""
import logging
from langchain_aws import ChatBedrock
from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ** CONFIGURE THIS **
# Get this from the Bedrock Knowledge Base console after you create it.
KNOWLEDGE_BASE_ID = "YOUR_KNOWLEDGE_BASE_ID"  # <-- IMPORTANT: e.g., "ABC123XYZ"
MODEL_1_ANALYST_ARN = "arn:aws:bedrock:us-east-1:982945613320:application-inference-profile/wwshs11qcgkg3"  # Claude 3.5 Sonnet


def main():
    """Sets up the RAG chain and starts an interactive query session."""
    if KNOWLEDGE_BASE_ID == "YOUR_KNOWLEDGE_BASE_ID":
        print("Error: Please update the KNOWLEDGE_BASE_ID in the script.")
        return

    logger.info("Initializing models and retriever...")

    # Initialize the LLM for generating answers
    llm = ChatBedrock(
        model_id=MODEL_1_ANALYST_ARN,
        model_kwargs={"temperature": 0.2, "anthropic_version": "bedrock-2023-05-31"}
    )

    # Initialize the retriever for the Bedrock Knowledge Base
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}  # Retrieve top 5 chunks
    )

    # Create the Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\n\n" + "=" * 80)
    print("      KNOWLEDGE BASE IS READY. ASK QUESTIONS ABOUT YOUR PROJECT:")
    print("=" * 80 + "\n")
    print("Type 'exit' to quit.\n")

    # Interactive query loop
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        logger.info(f"Invoking RAG chain with query: '{query}'")

        try:
            result = qa_chain.invoke({"query": query})

            print("\n--- Answer ---\n")
            print(result['result'])
            print("\n--- Sources ---\n")
            for i, doc in enumerate(result['source_documents']):
                print(
                    f"Source {i + 1} (from s3://{doc.metadata['location']['s3Location']['uri']}):\n{doc.page_content[:250]}...\n")
        except Exception as e:
            logger.error(f"An error occurred during query: {e}")
            print("\nAn error occurred. Please check the logs and ensure your Knowledge Base is synced.")


if __name__ == "__main__":
    main()