# Project working rules

- Read RESEARCH_PLAN.md before working on the research pipeline.
- Do not run training, inference, experimental preprocessing, or evaluation on the user's local computer. Experiments require a user-designated remote environment.
- Create a separate task branch before changing code or project documentation. Use feat/<task>, fix/<task>, or chore/<task>. Do not commit changes directly to main after the initial source import.
- Push each completed task branch and create a pull request against main so the user can inspect the diff. Include what changed, validation performed, and untested behavior.
- Leave pull requests unmerged unless the user explicitly asks to merge. Avoid force pushes and preserve unrelated work.
- Keep datasets, model weights, generated results, credentials, and local environment files out of Git. Preserve upstream license and attribution.
- Source review and non-experimental static checks are allowed locally. Do not report experiments or model quality as verified unless actually measured in the designated environment.
