# Contributing to ActiMotus

We'd love you to contribute to ActiMotus!

Questions, feature requests, and bug reports are all welcome as [GitHub Discussions or Issues](https://github.com/actimotus/actimotus/issues).

## Pull Requests
Since the project is currently a work-in-progress, we are figuring out the best workflows as we go. If you submit a Pull Request, we will review it and guide you through any changes needed. Don't worry about getting it perfect the first time—let's build this together!

### Prerequisites
- **Python 3.11** or higher.
- **uv** for dependency and environment management.
- **git** for version control.

### Installation and Setup
1. **Fork and Clone.** Fork the repository on GitHub and clone your fork locally:
```bash
git clone git@github.com:<your-username>/actimotus.git
cd actimotus
```

2. **Install uv.** If you don't have it yet, install uv following the [official guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Install Dependencies.** Run the sync command to create the virtual environment and install all dependencies (including dev, test, and doc tools):
```bash
uv sync
```

4. **Check out a new branch.** Create a new branch for your changes:
```bash
git checkout -b my-new-feature-branch
# Make your changes...
```

5. **Run Tests and Linting.** Please run the checks locally to ensure the code is formatted correctly.
```bash
# Run pre-commit checks on all files
uv run pre-commit
```
