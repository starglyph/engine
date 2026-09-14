# Agent Instructions (AGENTS)

**starglyph** simulates star fields and recognizes sky images, with constellation and object overlays. Code lives in the `prototype/` Rust workspace: engines, CLIs, an HTTP service, and a Tauri 2 desktop application with HTML/CSS/vanilla JS.

## Where to Find Context

- [README.md](README.md) describes current components and quick-start commands; [docs/README.md](docs/README.md) indexes the documentation.
- Use [docs/app.md](docs/app.md) for the desktop application, [docs/serve.md](docs/serve.md) for the HTTP API, and [docs/data-sources.md](docs/data-sources.md) for data. Read what the task needs.
- `docs/stack.md` describes the initial prototype choices; check README and `prototype/Cargo.toml` for the current stack.

## Code Principles

- Keep two explicit subsystems: the **simulator** (data, rendering, noise) and the **recognizer** (matching, overlays). Separate their interfaces and artifacts (datasets, weights, configurations) where practical.
- Document public catalogs and data formats in `docs/` as they are introduced.
- Avoid combining ground-truth generation (star catalog, camera) and recognition heuristics in one module unless necessary.

## Documentation Language

- Conceptual documents in `docs/` may be in **Russian**, as they are now. Follow team conventions for technical API/CI documentation; when conventions differ, prioritize clarity for the repository's readers.

## When to Ask the User

- When adding data, first check provenance and licensing in `docs/data-sources.md`, `THIRD_PARTY_LICENSES.md`, and `ATTRIBUTION.md`; ask only about rights and usage terms that are not already documented.
- Clarify the target mobile platform and performance constraints when the task involves the mobile client.

## Public Documentation

- This repository is **public**. Do **not mention** private or restricted repositories in commits, PRs, `docs/`, `openspec/`, or any other files added to Git. This includes their names, organizations, URLs, and artifact paths.
- The product backlog and strategic notes are maintained outside this repository. Keep documentation here focused on code, public specifications (`openspec/specs/`), and materials in `docs/`.
- OpenSpec in `openspec/` describes **implementation requirements** for this repository; do not reference external task trackers or private documents.

## Agent Workflows

- Managed skills live in `.agents/skills/`; their versions and source are pinned in `skills.lock.yaml`.
- Working environments are Codex (Sol or Astra) and Cursor. By default, the current agent performs the task directly in the user's chosen environment and model. Activate `task-delegation` and orchestration only on explicit request; an installed skill alone does not require Composer, Terra, or a model switch.
- The primary orchestration skill is `codex-orchestration`, explicitly included in `skills.lock.yaml`. The parent stays on the current session model; child-role routing is defined in [model-routing.md](.agents/skills/codex-orchestration/references/model-routing.md).
- Use `work-intake` for underspecified tasks; perform clear local changes directly. Do not create an OpenSpec change or a full change brief solely to satisfy a procedure.
- Explicit user requests take precedence over procedural skill restrictions. When both planning and implementation are requested, continue after the plan without asking for permission again unless a material question remains unresolved. Requests limited to analysis or proposals stay within those boundaries.
- For OpenSpec changes, respect the selected host and canonical artifact store. Cursor commands and skills live in `.cursor/`; their presence alone does not make them available in Codex.
- Select skills for the specific task. Use `improve` for a dedicated audit and plans, and design skills for relevant visual work. shadcn/Tailwind recommendations do not change the existing vanilla JS stack. Use `modern-web-guidance` to select or verify web APIs and compatibility, without requiring a network call for every cosmetic edit.
- Select checks based on the affected behavior. After they pass, broaden or repeat them only for new changes, failures, or unresolved risks. Run Cargo from `prototype/`; rebuild and relaunch the Tauri application when verifying changes to its embedded UI.
- If a skill instruction causes a pause or additional approval request, identify the exact file, instruction, and reason. An unavailable supporting tool does not block independent work.
- Keep project-specific overrides here or in a local skill outside the lock file; do not hand-edit managed copies. Verify the installation with `agentmem skills verify` and preview updates with `agentmem skills pull --dry-run`.

<!-- agentmem:closeout:start -->
This repository is registered in agentmem as `starglyph/engine`.
Run `@closeout for starglyph/engine` after non-trivial work (skill: `.agents/skills/closeout/SKILL.md`).
Consult `.agents/skills/agent-memory-usage/SKILL.md` for MCP usage.
<!-- agentmem:closeout:end -->

For this project, retrieve memory context once per substantive task. If MCP is unavailable, report it briefly and continue using repository context. Run closeout when the task is complete. In Codex, use the current conversation unless its transcript path is provided; do not search unrelated Cursor sessions. Record only useful decisions and lessons, concisely and without a 500–800-word target. If there are no such lessons, no event or memory draft is required.
