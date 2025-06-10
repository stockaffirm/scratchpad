# state.py
from typing import TypedDict, List, Dict, Any


class SchemaAnalysisState(TypedDict):
    input_schema_list: List[str]  # User-provided list of schemas
    all_db_schemas: List[str] = None  # All schemas from the DB (mocked initially)

    # For iterative processing of one input schema against all db schemas
    current_input_schema_idx: int = 0  # Index for input_schema_list
    current_db_schema_idx: int = 0  # Index for all_db_schemas

    # Normalization and comparison results
    canonical_forms: Dict[str, str] = {}  # Cache: original_name -> canonical_name
    # Feedback and advanced scoring will be added later
    # feedback_rules: Dict[tuple, Any] = {}
    # comparison_cache: Dict[tuple, float] = {}

    final_output: Dict[str, Dict[str, float]] = {}  # The final JSON structure