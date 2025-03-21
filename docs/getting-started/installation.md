# Installation Guide

## Prerequisites

Before you begin, ensure you have:

- A Unix-like operating system (Linux, macOS) or Windows (not recommended)
- [Rye](https://rye-up.com/) installed on your system

## Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://gitlab.fhnw.ch/ipole-bat/projection
   cd projection
   ```

2. Sync the environment to install all dependencies:
   ```bash
   rye sync --no-lock
   ```

3. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Add required API keys for logging:
     ```
     NEPTUNE_API_TOKEN=your_neptune_token
     WANDB_API_KEY=your_wandb_token
     ```

4. Activate the environment:
   ```bash
   . .venv/bin/activate
   ```

## Verifying Installation

To verify your installation:

1. Run the test suite:
   ```bash
   rye run test
   ```


## Troubleshooting

If you encounter any issues:

1. Make sure Rye is properly installed
2. Check that all dependencies are installed correctly
3. Verify your Python version (requires >= 3.12)
4. Ensure all environment variables are set correctly