#include "tokenizer.hpp"

#include <sentencepiece_processor.h>
#include <stdexcept>
#include <filesystem>

namespace translation {

namespace fs = std::filesystem;

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor src_sp;
    sentencepiece::SentencePieceProcessor tgt_sp;

    explicit Impl(const std::string& spm_dir) {
        // MarianMT opus-mt-en-fr ships source.spm + target.spm.
        // If the model uses a shared vocab, both paths point to the same file —
        // just symlink or duplicate it; the code stays identical either way.
        const fs::path dir{spm_dir};
        const std::string src_path = (dir / "source.spm").string();
        const std::string tgt_path = (dir / "target.spm").string();

        if (const auto s = src_sp.Load(src_path); !s.ok())
            throw std::runtime_error("SPM source load failed [" + src_path + "]: " + s.ToString());

        if (const auto s = tgt_sp.Load(tgt_path); !s.ok())
            throw std::runtime_error("SPM target load failed [" + tgt_path + "]: " + s.ToString());
    }
};

Tokenizer::Tokenizer(const std::string& spm_dir)
    : impl_(std::make_unique<Impl>(spm_dir)) {}

Tokenizer::~Tokenizer() = default;

std::vector<std::vector<std::string>>
Tokenizer::encode_batch(const std::vector<std::string>& texts) const {
    std::vector<std::vector<std::string>> out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        std::vector<std::string> tokens;
        impl_->src_sp.Encode(t, &tokens);
        out.push_back(std::move(tokens));
    }
    return out;
}

std::vector<std::string>
Tokenizer::decode_batch(const std::vector<std::vector<std::string>>& batches) const {
    std::vector<std::string> out;
    out.reserve(batches.size());
    for (const auto& tokens : batches) {
        std::string text;
        impl_->tgt_sp.Decode(tokens, &text);
        out.push_back(std::move(text));
    }
    return out;
}

} // namespace translation