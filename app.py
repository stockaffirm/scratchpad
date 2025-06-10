# app.py
from langgraph.graph import StateGraph, END
from state import SchemaAnalysisState  # If in separate files
from graph_nodes import (
    fetch_db_schemas_node,
    get_canonical_form_node,
    compare_schemas_node,
    advance_or_end_node
)
import json

# Define the workflow
workflow = StateGraph(SchemaAnalysisState)

# Add nodes
workflow.add_node("fetch_db_schemas", fetch_db_schemas_node)
workflow.add_node("get_canonical", get_canonical_form_node)
workflow.add_node("compare", compare_schemas_node)
workflow.add_node("advance_or_end", advance_or_end_node)

# Set entry point
workflow.set_entry_point("fetch_db_schemas")

# Define edges
workflow.add_edge("fetch_db_schemas", "get_canonical")  # Start first comparison
workflow.add_edge("get_canonical", "compare")
workflow.add_edge("compare", "advance_or_end")


# Conditional edge after advancing
def should_continue_processing(state: SchemaAnalysisState):
    if state['current_input_schema_idx'] >= len(state['input_schema_list']):
        print("CONDITION: All input schemas processed. ENDING.")
        return "end_processing"  # Go to END
    else:
        print(
            f"CONDITION: Continue processing for input: {state['input_schema_list'][state['current_input_schema_idx']]}, db_idx: {state['current_db_schema_idx']}")
        return "continue_comparison"  # Loop back to get_canonical for next pair


workflow.add_conditional_edges(
    "advance_or_end",
    should_continue_processing,
    {
        "continue_comparison": "get_canonical",
        "end_processing": END,
    }
)

# Compile the graph
app = workflow.compile()

# --- Run the agent ---
if __name__ == "__main__":
    initial_input_schemas = ["ADMDEV", "USER_UAT", "NONEXISTENT_CORE"]

    initial_state = SchemaAnalysisState(
        input_schema_list=initial_input_schemas,
        # Initialize other fields as per TypedDict defaults or explicitly if needed
        all_db_schemas=[],  # Will be populated by fetch_db_schemas_node
        current_input_schema_idx=0,
        current_db_schema_idx=0,
        canonical_forms={},
        final_output={}
    )

    print("--- Invoking Agent ---")
    # For streaming events and seeing intermediate states:
    # for event in app.stream(initial_state, {"recursion_limit": 100}):
    #     for key, value in event.items():
    #         print(f"--- Event from node: {key} ---")
    #         print(json.dumps(value, indent=2))
    #     print("\n")

    final_state = app.invoke(initial_state, {"recursion_limit": 150})  # Increased recursion limit

    print("\n--- Final Output ---")
    print(json.dumps(final_state['final_output'], indent=2))

    print("\n--- Canonical Forms Cache ---")
    print(json.dumps(final_state['canonical_forms'], indent=2))