# Project-specific agent instructions

These rules record explicit behavior corrections made by the user while working on this repository.

## Conversation continuity

- Treat brief clarification questions during active work as side questions by default. Answer them, then continue the authorized task without waiting for another prompt.
- Do not stop an implementation merely because the user asks what is being changed. Stop only when the user explicitly pauses, replaces, or cancels the task, or when a real decision or permission is required.

## Language preference

- The user often writes requests in English, but prefers responses in natural Traditional Chinese (`繁體中文`) by default.
- Keep code, identifiers, commands, filenames, and technical terms in English when translating them would reduce precision.

## Git conventions

- Do not use the `codex/` prefix for branches in this repository.
- Use a short, descriptive branch name such as `benchmark-foundation`.
- For commit messages, the format would be `<type>: <short description>`, where `<type>` is one of `feat`, `fix`, `refactor`, `test`, or `docs`.
- I prefer to put docs inside the last commit of a feature branch, rather than in a separate commit, unless the docs are a major addition or change.
- I would often use `git commit --amend` to update the last commit message or content when it is not pushed yet, rather than creating a new commit for small changes.

## Graphify freshness

- If graphify output is stale, still points at retired or legacy paths, or starts obstructing accurate analysis, update the graph before relying on it.
- Treat current source files as authoritative while an update is pending.

## Testing restraint

- Do not add persistent tests reflexively for every change.
- Add a test only when it protects an important behavior or contract, covers a meaningful regression risk, or the user explicitly requests it.
- Prefer existing checks and avoid tests that merely duplicate current coverage or assert implementation details without practical value.

## Markdown formatting

- Write each Markdown paragraph and each individual list item on one physical line.
- Do not insert manual line breaks in the middle of a sentence; let the editor or IDE handle visual wrapping.
- Code blocks and Markdown structures that inherently require multiple lines, such as tables, are exempt.

## Running conventions
- Smoke tests are allowed, but ensure they cover only the most important behaviors and do not take too long to run.
- Control group aren't mandatory as faster iteration on model with shorter train time and higher accuracy is preferred.

## Local machine notes
Check if the `AGENTS.local.md` file exists for machine-specific instructions. It is intentionally excluded from Git and can be refreshed when the machine changes.