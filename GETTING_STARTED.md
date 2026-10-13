# Getting Started with DriftGate

This guide will walk you through the initial setup of DriftGate and how to run its powerful suite of tests against an example multi-agent application.

## 1. Installation

First, ensure you have Python 3.11+ and `uv` installed. Then, clone this repository and install the dependencies.

```bash
# Clone the repository
git clone <your-repo-url>
cd genai-api-quality-gates

# Install dependencies using uv
uv sync --extra dev
```

## 2. Initialize the Framework

DriftGate comes with a powerful `init` command that scaffolds a configuration file for you.

```bash
# Make sure your virtual environment is active
uv run driftgate init
```

This command creates `driftgate.config.yaml` in your project root. This file is the control panel for the entire framework.

## 3. Configure the LLM Judge

For this tutorial, we'll use a local LLM running via [Ollama](https://ollama.com/) to act as our intelligent "judge" for regression tests.

1.  **Install and Run Ollama:** Follow the instructions on the Ollama website to install it and pull a model.

    ```bash
    # We recommend using the new Phi-3 model for a good balance of size and performance
    ollama pull phi3
    ```

2.  **Configure DriftGate:** Open the `driftgate.config.yaml` file. It is already configured to use Ollama and the `phi3` model by default.

    ```yaml
    # driftgate.config.yaml

    judge:
      # Provider for the judge model. Supported: "ollama", "anthropic", "openai", "google"
      provider: "ollama"

      ollama:
        # The model to use for judging (e.g., "llama3", "phi3")
        model: "phi3"
        # The base URL of your Ollama server
        base_url: "http://localhost:11434"
        # Timeout in seconds
        timeout: 30
    ```

## 4. Run the Example Tests

The framework includes a mock multi-agent application and a full suite of example tests. You can run them with a single command.

```bash
# Run all tests in the examples directory
uv run pytest examples/tests/
```

This command will execute several types of tests against the mock service:

*   **Golden Dataset Regressions:** Runs the prompts from `golden_multi_agent.jsonl` through the service and uses your local Ollama model to judge if the new responses are semantically equivalent to the approved "golden" ones.
*   **Agent Contract Tests:** Validates that the `MathAgent`'s output conforms to the schema and tool-use rules defined in its `agent-contract.yaml`.
*   **Red Team Attacks:** Injects a library of adversarial prompts to ensure the service responds safely and does not get hijacked.

## 5. Explore the Results

After the test run, you can find detailed reports and artifacts in the `.driftgate/` directory. This includes a markdown report (`report.md`) that is perfect for viewing in CI/CD systems.

## Next Steps

You have now successfully run a comprehensive AI quality assurance suite! You can now:

*   Explore the `examples/` directory to see how the mock service and tests are implemented.
*   Read the **`USAGE_GUIDE.md`** file to learn how to apply these features to your own projects.
*   Use the `uv run driftgate dataset approve` command to build your own golden datasets from recorded API traffic.
