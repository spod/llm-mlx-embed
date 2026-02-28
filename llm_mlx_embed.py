import json
import llm
import mlx.core as mx
import mlx_lm


@llm.hookimpl
def register_embedding_models(register):
    models_file = llm.user_dir() / "llm-mlx.json"
    if not models_file.exists():
        return
    models = json.loads(models_file.read_text())
    for model_id, config in models.items():
        if "embedding" not in model_id.lower():
            continue
        # Use the repo's short name (after the last /) as a convenience alias
        short_name = model_id.split("/")[-1]
        aliases = list(config.get("aliases", [])) or [short_name]
        register(MlxEmbeddingModel(model_id), aliases=aliases)


class MlxEmbeddingModel(llm.EmbeddingModel):
    def __init__(self, model_id):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            self._model, self._tokenizer = mlx_lm.load(self.model_id)
        return self._model, self._tokenizer

    def embed_batch(self, texts):
        model, tokenizer = self._load()
        results = []
        for text in texts:
            tokens = tokenizer.encode(text)
            token_array = mx.array([tokens])           # [1, seq_len]
            hidden = model.model(token_array)          # [1, seq_len, hidden_size]
            vec = hidden[0, -1, :]                     # last-token pool → [hidden_size]
            vec = vec / mx.sqrt(mx.sum(vec * vec))     # L2 normalise
            mx.eval(vec)
            results.append(vec.tolist())
        return iter(results)
