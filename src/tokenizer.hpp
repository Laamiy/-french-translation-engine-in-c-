#pragma once

#include <string>
#include <vector>
#include <memory>

namespace translation {

// Pimpl hides sentencepiece headers from all consumers.
// Encode/decode are const and fully thread-safe after construction.
class Tokenizer {
public:
    explicit Tokenizer(const std::string& spm_dir);
    ~Tokenizer();

    Tokenizer(const Tokenizer&)            = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Thread-safe: may be called concurrently from any thread.
    [[nodiscard]] std::vector<std::vector<std::string>>
    encode_batch(const std::vector<std::string>& texts) const;

    [[nodiscard]] std::vector<std::string>
    decode_batch(const std::vector<std::vector<std::string>>& token_batches) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace translation