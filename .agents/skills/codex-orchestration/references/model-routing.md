# Model routing

Use this reference for every collaboration subagent started by
**`codex-orchestration`**.

## Session-aware roles

The parent stays on the current session model (for example Astra or Sol).
Never switch or re-create it to satisfy a default. Resolve `parent_model` from
runtime session metadata; do not infer an exact model ID from prose.
Explicit user role assignments override the defaults below. Project role
defaults apply next; this table supplies the remaining defaults.

| Role | Model | Effort | Context |
|---|---|---|---|
| Design and OpenSpec authoring | parent model | `xhigh` | fresh |
| Read-only fact gathering | `gpt-5.6-terra` | `medium` | fresh |
| Implementation slice | `gpt-5.6-terra` | `high` | fresh |
| Profile QA | `gpt-5.6-terra` | `high` | fresh |
| Stuck implementation escalation | parent model | `high` | fresh |
| Independent review | parent model | `xhigh` | fresh |

Reserve `max` for an explicitly justified, high-risk follow-up review. Do
not raise effort merely because a task is large; first improve semantic
boundaries and acceptance evidence.

## Spawn contract

Model inheritance and conversation inheritance are separate. For a parent-model
role, omit `model` when the runtime supports inheriting the model with
`fork_turns: "none"`; otherwise pass the runtime-reported exact parent model ID.
Do not use a literal `inherit` model ID unless the tool explicitly supports it.
If no model ID is exposed but isolated model inheritance is supported, use that
mechanism without asking the user to repeat their selection. Report the model
as session-inherited when its exact ID is unavailable.

Every `spawn_agent` call includes:

- the resolved role model (explicit or inherited as above) and supported effort;
- `fork_turns: "none"` for role isolation, including inherited-model roles;
- a unique lowercase task name;
- a self-contained bounded prompt with files/artifacts to read, outcome,
  non-goals, permissions, acceptance, expected report, and stopping conditions;
- a reminder that the working tree is shared and nested delegation is not
  allowed.

Do not rely on inherited conversation context. Pass only the relevant OpenSpec
context paths and project evidence from disk.

Use `followup_task` only for the single correction retry of the same semantic
slice. Never use it to turn an implementer into a reviewer. Review and parent-model
escalation always use a fresh `spawn_agent` call.

Use `wait_agent` with a long bounded wait instead of frequent polling. Check
available slots when necessary before spawning.

## Concurrency

The parent occupies one slot. Parallelize only independent read-only work. Code
edits, migrations, generated-file updates, task checkboxes, QA fixture
mutations, main-spec sync, archive, and review repairs are sequential.

Implementation and review agents may not spawn their own subagents. Keep role
and write ownership visible to the parent.

## Failure policy

If a required resolved role cannot start, report the role and runtime error.
Use an already authorized alternative when available; otherwise report
`BLOCKED@runtime` for the dependent work. Do not silently substitute models or
make the parent an implementation fallback. An unavailable unused model is
not a blocker: an Astra-parent run does not require Sol.

For failed slice acceptance:

1. one correction retry on the same Terra agent;
2. one fresh parent-model `high` escalation only when requirements and environment are
   sound and stronger implementation reasoning is plausibly useful;
3. rerun acceptance and stop when it still fails.

Profile-QA and review loops are also finite: one Terra repair batch and one
rerun or re-review. Record every retry and escalation in the final Run field.
