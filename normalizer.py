# normalizer.py
import re


class SimpleSchemaNormalizer:
    def __init__(self):
        # Initial, basic rules. Expand these significantly!
        # (Canonical_Core, [Variations]), (Canonical_Env, [Variations])
        self.core_synonyms = {
            "admin": ["admin", "adm"],
            "user": ["user", "usr"],
        }
        self.env_synonyms = {
            "prod": ["prod", "prd", ""],  # "" means if only core is found, assume prod
            "dev": ["dev", "develop"],
            "uat": ["uat"],
            "test": ["test", "tst"],
        }

        # Pre-compile for faster lookup: variation -> canonical_core/env
        self.core_map = {var: canon for canon, var_list in self.core_synonyms.items() for var in var_list}
        self.env_map = {var: canon for canon, var_list in self.env_synonyms.items() for var in var_list}

        # Sort by length for greedy matching (longest first)
        self.sorted_core_keys = sorted(self.core_map.keys(), key=len, reverse=True)
        self.sorted_env_keys = sorted(self.env_map.keys(), key=len, reverse=True)

    def normalize(self, schema_name: str) -> str:
        s_lower = schema_name.lower()

        found_core = None
        found_env = None

        # Attempt to split by common delimiters
        parts = re.split(r'[_,-]', s_lower)

        # Try to identify core and env from parts
        # This is a very basic identification, assumes one core and one env part
        remaining_parts = list(parts)  # Make a copy to modify

        for part_idx, part in enumerate(remaining_parts):
            for core_key in self.sorted_core_keys:
                if part == core_key:
                    found_core = self.core_map[core_key]
                    remaining_parts[part_idx] = None  # Mark as used
                    break
            if found_core:
                break

        for part_idx, part in enumerate(remaining_parts):
            if part is None: continue
            for env_key in self.sorted_env_keys:
                if part == env_key:
                    found_env = self.env_map[env_key]
                    remaining_parts[part_idx] = None  # Mark as used
                    break
            if found_env:
                break

        # If no clear parts, try substring matching on the whole name (for names like "adminprd")
        if not found_core:
            for core_key in self.sorted_core_keys:
                if core_key in s_lower:
                    found_core = self.core_map[core_key]
                    # Simplistic: assume rest might be env or nothing
                    break

        if not found_env:
            # Search in the original string, but try not to overlap with found_core if possible
            search_string_for_env = s_lower
            if found_core:  # Try to remove core part to avoid confusion
                temp_search = s_lower.replace(found_core, "")  # Simple removal
                if temp_search:  # if something remains
                    search_string_for_env = temp_search

            for env_key in self.sorted_env_keys:
                if env_key == "": continue  # Skip empty string for substring search
                if env_key in search_string_for_env:
                    found_env = self.env_map[env_key]
                    break

        # Construct canonical name: core_prod
        if found_core:
            # Your rule: always map to the 'prod' version for grouping
            return f"{found_core}_prod"

        # Fallback: if no core found, return a simplified version of original
        # This fallback needs to be robust or signal an issue.
        return re.sub(r'[^a-z0-9]', '', s_lower)  # Basic alphanumeric cleanup


# Initialize a global normalizer instance for easy access by nodes
schema_normalizer_instance = SimpleSchemaNormalizer()