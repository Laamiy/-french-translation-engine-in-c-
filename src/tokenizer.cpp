#include "tokenizer.hpp"
#include <sentencepiece_processor.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace translation 
{
    namespace fs = std::filesystem;
    using json = nlohmann::json;

    struct Tokenizer::Impl 
    {
        sentencepiece::SentencePieceProcessor src_sp;
        sentencepiece::SentencePieceProcessor tgt_sp;
        std::unordered_map<std::string, int32_t> src_vocab; // piece -> id
        std::unordered_map<int32_t, std::string> tgt_vocab; // id -> piece

        explicit Impl(const std::string& spm_dir) 
        {
            const fs::path dir{spm_dir};

            const std::string src_path = (dir / "source.spm").string();
            const std::string tgt_path = (dir / "target.spm").string();

            if (const auto s = src_sp.Load(src_path); !s.ok())
                throw std::runtime_error("SPM source load failed: " + s.ToString());

            if (const auto s = tgt_sp.Load(tgt_path); !s.ok())
                throw std::runtime_error("SPM target load failed: " + s.ToString());

            // Load vocab.json — piece->id mapping, identical to what AutoTokenizer uses.
            // but SPM's internal IDs differ from HuggingFace's vocab.json IDs.

            const std::string vocab_path = (dir / "vocab.json").string();
            std::ifstream vocab_file(vocab_path);

            if (!vocab_file)
                throw std::runtime_error("vocab.json not found: " + vocab_path);

            json vocab_json;
            vocab_file >> vocab_json;

            for (const auto& [piece, id] : vocab_json.items())
                src_vocab[piece] = id.get<int32_t>();

            // Build reverse map for decoding
            for (const auto& [piece, id] : src_vocab)
                tgt_vocab[id] = piece;
        }

        // Encode using SPM for subword splitting, then remap to vocab.json IDs.
        std::vector<int32_t> encode(const std::string& text) const 
        {
            std::vector<std::string> pieces;
            src_sp.Encode(text, &pieces);

            std::vector<int32_t> ids;
            ids.reserve(pieces.size() + 1);

            for (const auto& piece : pieces) 
            {
                auto it = src_vocab.find(piece);

                if (it != src_vocab.end())
                    ids.push_back(it->second);
                else
                    ids.push_back(src_vocab.count("<unk>") ? src_vocab.at("<unk>") : 0);
            }
            // Append EOS token — same as AutoTokenizer  for MarianMT
            ids.push_back(0);
            return ids;
        }

        // Decode: map IDs back to pieces via reverse vocab, then SPM decode.
        std::string decode(const std::vector<int32_t>& ids) const 
        {
            std::vector<std::string> pieces;
            pieces.reserve(ids.size());
            
            for (auto id : ids) 
            {
                if (id == EOF_TOKEN || id == PAD_TOKEN) 
                    continue; // skip EOS, PAD
                auto it = tgt_vocab.find(id);

                if (it != tgt_vocab.end())
                    pieces.push_back(it->second);
            }

            std::string text;
            tgt_sp.Decode(pieces, &text);
            return text;
        }
    };

    Tokenizer::Tokenizer(const std::string& spm_dir) : impl_(std::make_unique<Impl>(spm_dir)) 
    {}

    Tokenizer::~Tokenizer() = default;

    std::vector<std::vector<int32_t>>
    Tokenizer::encode_ids_batch(const std::vector<std::string>& texts) const 
    {
        std::vector<std::vector<int32_t>> out;
        out.reserve(texts.size());
        for (const auto& t : texts)
            out.push_back(impl_->encode(t));
        return out;
    }

    std::vector<std::string>
    Tokenizer::decode_ids_batch(const std::vector<std::vector<int32_t>>& id_batches) const 
    {
        std::vector<std::string> out;
        out.reserve(id_batches.size());
        for (const auto& ids : id_batches)
            out.push_back(impl_->decode(ids));
        return out;
    }

    // String-based encode/decode (unused in main path but kept for completeness)
    std::vector<std::vector<std::string>>
    Tokenizer::encode_batch(const std::vector<std::string>& texts) const 
    {
        std::vector<std::vector<std::string>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) {
            std::vector<std::string> pieces;
            impl_->src_sp.Encode(t, &pieces);
            out.push_back(std::move(pieces));
        }
        return out;
    }

    std::vector<std::string>
    Tokenizer::decode_batch(const std::vector<std::vector<std::string>>& batches) const 
    {
        std::vector<std::string> out;
        out.reserve(batches.size());
        for (const auto& pieces : batches) {
            std::string text;
            impl_->tgt_sp.Decode(pieces, &text);
            out.push_back(std::move(text));
        }
        return out;
    }

} // namespace translation