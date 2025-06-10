# graph_nodes.py (or in your main app file)
from .state import SchemaAnalysisState  # Assuming state.py is in the same directory
from .normalizer import schema_normalizer_instance  # Assuming normalizer.py


def fetch_db_schemas_node(state: SchemaAnalysisState) -> SchemaAnalysisState:
    print("NODE: Fetching DB Schemas (Mocked)")
    # In a real scenario, this connects to the DB
    # For now, mock the data:
    state['all_db_schemas'] = ["ADMPRD", "ADMDEV", "USER_PROD", "USER_UAT", "DEV_ADM", "ADMINPRD", "XYZ_TEST", "APP_V1"]
    state['current_input_schema_idx'] = 0
    state['current_db_schema_idx'] = 0
    if state['input_schema_list']:  # Initialize output dict for the first input schema
        first_input_schema = state['input_schema_list'][0]
        state['final_output'][first_input_schema] = {}
    return state


def get_canonical_form_node(state: SchemaAnalysisState) -> SchemaAnalysisState:
    input_schema_orig = state['input_schema_list'][state['current_input_schema_idx']]
    db_schema_orig = state['all_db_schemas'][state['current_db_schema_idx']]

    print(f"NODE: Normalizing - Input: {input_schema_orig}, DB: {db_schema_orig}")

    if input_schema_orig not in state['canonical_forms']:
        state['canonical_forms'][input_schema_orig] = schema_normalizer_instance.normalize(input_schema_orig)
    if db_schema_orig not in state['canonical_forms']:
        state['canonical_forms'][db_schema_orig] = schema_normalizer_instance.normalize(db_schema_orig)

    print(f"  Canonical Input: {state['canonical_forms'][input_schema_orig]}")
    print(f"  Canonical DB:    {state['canonical_forms'][db_schema_orig]}")
    return state


def compare_schemas_node(state: SchemaAnalysisState) -> SchemaAnalysisState:
    input_schema_orig = state['input_schema_list'][state['current_input_schema_idx']]
    db_schema_orig = state['all_db_schemas'][state['current_db_schema_idx']]

    canonical_input = state['canonical_forms'][input_schema_orig]
    canonical_db = state['canonical_forms'][db_schema_orig]

    score = 0.0
    if canonical_input == canonical_db:
        score = 1.0

    print(f"NODE: Comparing {canonical_input} vs {canonical_db} -> Score: {score}")

    # Ensure the sub-dictionary for the current input schema exists
    if input_schema_orig not in state['final_output']:
        state['final_output'][input_schema_orig] = {}

    state['final_output'][input_schema_orig][db_schema_orig] = score
    return state


def advance_or_end_node(state: SchemaAnalysisState) -> SchemaAnalysisState:
    print("NODE: Advancing Pointers")
    # Try to advance db_schema_idx first
    state['current_db_schema_idx'] += 1

    # If all_db_schemas processed for current_input_schema
    if state['current_db_schema_idx'] >= len(state['all_db_schemas']):
        state['current_input_schema_idx'] += 1  # Move to next input_schema
        state['current_db_schema_idx'] = 0  # Reset db_schema_idx

        # If there's a next input schema, initialize its output dict
        if state['current_input_schema_idx'] < len(state['input_schema_list']):
            next_input_schema = state['input_schema_list'][state['current_input_schema_idx']]
            state['final_output'][next_input_schema] = {}

    return state