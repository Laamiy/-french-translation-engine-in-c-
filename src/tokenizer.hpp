#pragma once

#include <string>
#include <vector>
#include <memory>

#define EOF_TOKEN 0
#define PAD_TOKEN 59513

namespace translation 
{
    // Pimpl hides sentencepiece headers from all consumers.
    // Encode/decode are const and fully thread-safe after construction.
    class Tokenizer 
    {
        public:
            explicit Tokenizer(const std::string& spm_dir);
            ~Tokenizer();

            Tokenizer(const Tokenizer&)            = delete;
            Tokenizer& operator=(const Tokenizer&)   = delete;

            // Thread-safe:
            [[nodiscard]] std::vector<std::vector<std::string>>
            encode_batch(const std::vector<std::string>& texts) const;

            [[nodiscard]] std::vector<std::string>
            decode_batch(const std::vector<std::vector<std::string>>& token_batches) const;

            [[nodiscard]] std::vector<std::vector<int32_t>>
            encode_ids_batch(const std::vector<std::string>& texts) const;

            // Decode integer IDs to text
            [[nodiscard]] std::vector<std::string>
            decode_ids_batch(const std::vector<std::vector<int32_t>>& id_batches) const;

        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
        };
} // namespace translation