// SPDX-License-Identifier: GPL-2.0-only
//
// Greedy tokens and logit margins of llama.cpp, the oracle of issue 30.
//
// It reads a GGUF file and a list of prompts given as token ids, decodes
// STEPS tokens after each prompt on the CPU with one thread, and prints one JSON
// document: for every prompt and step the chosen token, the two best logits and
// the margin between them. The end of sequence token does not stop a run, so
// every prompt gives exactly STEPS tokens.
//
// It is built against the llama.h of the commit that is pinned in
// bench/gguf/README.md, and it uses the batch functions that header marks as
// deprecated (llama_batch_init, llama_decode), because they are the ones that
// header documents; the flag -Wno-deprecated-declarations is part of the build.
//
// Usage: llama_golden MODEL.gguf STEPS PROMPTS.txt
//   PROMPTS.txt has one prompt per line: the token ids separated by spaces.
//   Lines that are empty or start with '#' are skipped.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

struct Step {
    llama_token token;
    float top;
    float second;
};

// The index of the largest logit, the first one on a tie as in argmax of Rabbit.
llama_token argmax(const float * logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

float second_best(const float * logits, int n, int best) {
    float second = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (i != best && logits[i] > second) {
            second = logits[i];
        }
    }
    return second;
}

bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, int first_position) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) (first_position + (int) i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // Only the last token of the batch needs its logits.
        batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
    }
    int32_t status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return status == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf STEPS PROMPTS.txt\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const int steps = std::atoi(argv[2]);
    if (steps <= 0) {
        std::fprintf(stderr, "STEPS must be positive\n");
        return 2;
    }

    std::vector<std::vector<llama_token>> prompts;
    {
        std::ifstream file(argv[3]);
        if (!file) {
            std::fprintf(stderr, "cannot read %s\n", argv[3]);
            return 2;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream words(line);
            std::vector<llama_token> ids;
            long id;
            while (words >> id) {
                ids.push_back((llama_token) id);
            }
            if (!ids.empty()) {
                prompts.push_back(ids);
            }
        }
    }
    if (prompts.empty()) {
        std::fprintf(stderr, "no prompt in %s\n", argv[3]);
        return 2;
    }
    size_t longest = 0;
    for (const auto & p : prompts) {
        longest = std::max(longest, p.size());
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    // The plain CPU kernels: the repacked weight layouts of the extra buffer
    // types are other code paths, and the oracle should be the reference one.
    model_params.use_extra_bufts = false;
    model_params.check_tensors = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = (uint32_t) (longest + (size_t) steps + 8);
    context_params.n_batch = (uint32_t) (longest + 8);
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.offload_kqv = false;
    context_params.op_offload = false;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    std::printf("{\n  \"vocab\": %d,\n  \"steps\": %d,\n  \"prompts\": [\n", n_vocab, steps);
    for (size_t index = 0; index < prompts.size(); index++) {
        const auto & prompt = prompts[index];
        for (llama_token id : prompt) {
            if (id < 0 || id >= n_vocab) {
                std::fprintf(stderr, "token %d is outside the vocabulary of %d\n", id, n_vocab);
                return 1;
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<Step> run;
        std::vector<llama_token> pending = prompt;
        int position = 0;
        for (int step = 0; step < steps; step++) {
            if (!decode(ctx, pending, position)) {
                std::fprintf(stderr, "llama_decode failed at step %d of prompt %zu\n", step, index);
                return 1;
            }
            position += (int) pending.size();
            const float * logits = llama_get_logits_ith(ctx, (int32_t) pending.size() - 1);
            if (logits == nullptr) {
                std::fprintf(stderr, "no logits at step %d of prompt %zu\n", step, index);
                return 1;
            }
            llama_token best = argmax(logits, n_vocab);
            run.push_back({best, logits[best], second_best(logits, n_vocab, best)});
            pending = {best};
        }
        std::printf("    {\n      \"prompt_ids\": [");
        for (size_t i = 0; i < prompt.size(); i++) {
            std::printf("%s%d", i ? ", " : "", prompt[i]);
        }
        std::printf("],\n      \"tokens\": [");
        for (size_t i = 0; i < run.size(); i++) {
            std::printf("%s%d", i ? ", " : "", run[i].token);
        }
        std::printf("],\n      \"top\": [");
        for (size_t i = 0; i < run.size(); i++) {
            std::printf("%s%.9g", i ? ", " : "", run[i].top);
        }
        std::printf("],\n      \"second\": [");
        for (size_t i = 0; i < run.size(); i++) {
            std::printf("%s%.9g", i ? ", " : "", run[i].second);
        }
        std::printf("]\n    }%s\n", index + 1 < prompts.size() ? "," : "");
    }
    std::printf("  ]\n}\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
