# Archived Legacy Scripts

`legacy_scripts/` contains the complete standalone Python files as they existed
after the local working changes were integrated with `origin/main` and before
the `src/vlm` restructure began.

These files are historical references:

- active code must not import from `archive`;
- compatibility wrappers at the repository root must not execute archived code;
- fixes and new features belong in `src/vlm`;
- Git history and the active package remain authoritative.

The repository-root files with matching names are intentionally retained as
small compatibility wrappers for existing commands and Python imports.
