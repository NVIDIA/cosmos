# AGENTS.md — Cosmos

This is the landing repository for NVIDIA Cosmos: the README, the [model reference](docs/reference/models.md) and the [Cosmos Cookbook](COOKBOOK.md), an index of agent-runnable recipes. Code, recipes and agent skills live in [Cosmos Framework](https://nvda.ws/4zRE0Md): read its `AGENTS.md` and `.agents/skills/` before running or changing a recipe.

## Where things go

- New recipes, skills and code go to Cosmos Framework, not this repository.
- Model facts (sizes, inputs, outputs, generation settings) belong in `docs/reference/models.md`.
- The recipe catalog is `COOKBOOK.md`.

## Keep in sync

- The example-checkpoint table appears in both `README.md` (Models) and `docs/reference/models.md` (Family). Change its rows in both files in the same commit.
- `COOKBOOK.md`: one short name per recipe; industry tags (`ANY` `AV` `ROB` `INFRA` `MFG` `AG` `MED`) and GPU-budget tags (`S` `M` `L` `XL`) only from its legend; each URL once, in the link block at the bottom. The full rules are in the maintainer comment at the top of that file.
- README "What's new" entries go newest first, as `- **[Mon YYYY]** …`.

## Links

Link files in this repository with relative paths. A link to a branch (`/blob/<branch>/…`) breaks once the branch is merged and deleted.

## Pull requests

Target `main`, keep one logical change per pull request, and describe what changed and why.
