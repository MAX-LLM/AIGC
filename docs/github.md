# GitHub preparation

Run commands from the project root. Install dependencies and run the essential
tests as described in [README.md](../README.md).

Build a clean source archive:

```bash
python scripts/package_release.py
```

The default output is `.runtime/release/MND_agent_github_ready.zip`. Use
`--output path/to/new_archive.zip` for another destination. The archive contains
the agent, web interface, SPJ workflow, tests, documentation, and source reference
files. It excludes credentials, local configuration, cases, logos, generated
results, caches, and Git history.

Configure the application through exported environment variables, or create
your own ignored `.env` file. Keep credentials out of committed files. Case data
under `vignettes/` and `research/spj_v1/cases/` stays local; pass an external bank
with `--case-dir` after cloning or extracting a release.

Before publishing, inspect `git status --short`, `git diff --check`, and the
files selected for upload. Ignore rules do not remove already tracked files.
The GitHub workflow runs the six essential test modules on Python 3.11 and 3.12
without model credentials or local case data.
