## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management. The required dependencies are specified in `pyproject.toml`, with exact resolved versions recorded in `uv.lock`.

Clone the repository and enter the project directory.

```bash
git clone git@github.com:yinzi-xin/backPAC.git
cd backPAC
```

Install `uv` if it is not already installed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the project's virtual environment and install the dependencies from the lock file.

```bash
uv sync --dev
```

or, if running on Leiden University's ALICE cluster:

```bash
uv sync --group alice
```

This creates a `.venv` environment and installs the dependencies specified by `uv.lock`.

Activate the virtual environment if you want to run Python commands directly (not needed if using VS Code).

```bash
source .venv/bin/activate
```