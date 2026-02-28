# llm-mlx-embed

A plugin for [simonw/llm](https://github.com/simonw/llm) to use with [simonw/llm-mlx](https://github.com/simonw/llm-mlx) to register MLX embedding models for use with llm embeddings functionality.

## Sample usage
``` bash
# Install llm and llm-mlx (assumes uv)
$ uv pip install llm
$ uv pip install llm-mlx

# Download an mlx embedding model
$ llm mlx download-model mlx-community/Qwen3-Embedding-4B-4bit-DWQ
...

# list embedding models - no MlxModels will be listed
$ llm embed-models
...

# Install this plugin
$ llm install -e .
...

# list installed embedding models
$ llm embed-models
...
Output should include:
MlxEmbeddingModel: mlx-community/Qwen3-Embedding-4B-4bit-DWQ (aliases: Qwen3-Embedding-4B-4bit-DWQ)

# Use an mlx embedding model from cli
$ llm embed -m Qwen3-Embedding-4B-4bit-DWQ -c 'hello' -f hex
0000d3b90000d2bb0000023d0000cd3c000005bb00.....

# If you get an error you need to first set a default embed model
$ llm embed-models default Qwen3-Embedding-4B-4bit-DWQ

```

## TODO
- [ ] Document
- [ ] Add tests
- [ ] File issue with upstream to determine if this should be a feature in llm-mlx instead of an additional plugin.

## References
* https://llm.datasette.io/en/stable/plugins/tutorial-model-plugin.html
* https://llm.datasette.io/en/stable/embeddings/writing-plugins.html
