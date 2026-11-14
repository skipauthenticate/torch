// The tokenizer and chat template of a GGUF file, as files a Hugging Face `tokenizers` library and a Jinja renderer
// read: dir/tokenizer.json and dir/chat_template.jinja (docs/model-files.md, "The tokenizer").
//
// The GGUF carries the vocabulary, the token types, the merges and the template. It does not carry the normalizer,
// the pre-tokenizer's regular expression, the decoder or the added tokens' flags: those follow from the
// pre-tokenizer's name, and only `qwen35` is known here. Ids are laid out as normal tokens first (the BPE
// vocabulary), then the added tokens (control or user-defined), then unused padding ids, which are left out.
#pragma once

#include <filesystem>

#include "gguf/gguf.h"

namespace torch::gguf {

struct TokenizerCounts {
    size_t vocab;   // BPE tokens
    size_t added;   // added tokens
    size_t unused;  // padding ids left out
    size_t merges;
    size_t template_bytes;
};

// Writes dir/tokenizer.json and dir/chat_template.jinja (creating dir). Throws Error for a file whose tokenizer this
// cannot export: a model other than gpt2, a pre-tokenizer other than qwen35, missing keys, or ids out of the layout.
TokenizerCounts export_tokenizer(const File& f, const std::filesystem::path& dir);

}  // namespace torch::gguf
