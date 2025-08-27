#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h> 
#include <cstdlib>
#include <string>
#include <optional>

using namespace std;
using namespace std::chrono;

static int embedding_table_size = 1000000;
static const int embedding_dim = 128;
static const int input_size = 720;
static const int num_bags = 20;
static int prefetch_distance = 4;
static int prefetch_hint = _MM_HINT_T0;
static bool enable_software_prefetch = false;

static void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto get_val = [&](const string& key) -> optional<string> {
            string prefix = "--" + key + "=";
            if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
            return nullopt;
        };

        if (auto v = get_val("software-prefetch")) {
            string s = *v;
            enable_software_prefetch = (s == "on" || s == "true" || s == "1");
        } else if (auto v = get_val("embed-size")) {
            embedding_table_size = stoi(*v);
        } else if (auto v = get_val("prefetch-distance")) {
            prefetch_distance = stoi(*v);
        } else if (auto v = get_val("prefetch-level")) {
            string s = *v;
            if (s.find("T0") != string::npos) prefetch_hint = _MM_HINT_T0;
            else if (s.find("T1") != string::npos) prefetch_hint = _MM_HINT_T1;
            else if (s.find("T2") != string::npos) prefetch_hint = _MM_HINT_T2;
            else prefetch_hint = _MM_HINT_T0;
        }
    }
}

static inline void prefetch_with_hint(const void* ptr) {
    // Dispatch to constant hints to satisfy compiler requirements
    switch (prefetch_hint) {
        case _MM_HINT_T0:
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
            break;
        case _MM_HINT_T1:
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T1);
            break;
        case _MM_HINT_T2:
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T2);
            break;
        case _MM_HINT_NTA:
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
            break;
        default:
            _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
            break;
    }
}


int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

void printVector(const std::vector<std::vector<float>>& vec) {
    for (const auto& row : vec) {
        for (float val : row) {
            std::cout << val;
        }
    }

    cout << "\nDone\n";
}

long long run_with_prefetching(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> output;
    int op_at_a_time = prefetch_distance > 0 ? prefetch_distance : 4;

    for (size_t i = 0; i < offsets.size(); ++i) {

        // Potential prefetch of upcoming offset metadata if beneficial
        // if (i + 1 < offsets.size()) {
        //     _mm_prefetch((const char*)&offsets[i + 1], prefetch_hint);
        // }

        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {

            if (j + op_at_a_time < end_idx) {
                const float* pf_ptr = &embedding_table[input[j + op_at_a_time] * embedding_dim];
                prefetch_with_hint(pf_ptr);
            }

            const float* data_ptr = &embedding_table[input[j] * embedding_dim];

            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
        
    }    
    //-------------------------------------------------------------------------------------------------------------------------------------------

    // printVector(output);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.\n";

    return duration.count();
}

long long run_with_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> output;

    // 64b => 8B => 2 floats at a time
    // 128b => 16B => 4 floats at a time
    // 256b => 32B => 8 floats at a time
    // 512b => 64B => 16 floats at a time
    
    int op_at_a_time = 4; // Input for number of operations at a time

    int steps = embedding_dim / op_at_a_time;

    for(size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            const float* data_ptr = &embedding_table[input[j] * embedding_dim];
            
            for(int k = 0; k < steps; k++) {
               
                __m128 curr = _mm_loadu_ps(&data_ptr[k * op_at_a_time]);

                __m128 prev = _mm_loadu_ps(&bag_embedding[k*op_at_a_time]);

                __m128 sum = _mm_add_ps(curr, prev);

                _mm_storeu_ps(&bag_embedding[k*op_at_a_time], sum);
            }
        }

        // for(int k = 0; k < steps; k++) {
               
        //     __m128 prev = _mm_loadu_ps(&bag_embedding[k*op_at_a_time]);

        //     for (int j = start_idx; j < end_idx; ++j) {
        //         const float* data_ptr = &embedding_table[input[j] * embedding_dim];
        //         __m128 curr = _mm_loadu_ps(&data_ptr[k * op_at_a_time]);
        //         prev = _mm_add_ps(curr, prev);
        //     }
            
        //     _mm_storeu_ps(&bag_embedding[k*op_at_a_time], prev);
        // }

        output.push_back(bag_embedding);
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------
    // printVector(output);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching and SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}


long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    // flush()
    
    // printVector(output);

    // cout << "Naive Over" << endl;

    return duration.count();

}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    // Prepare embedding table
    vector<float> embedding_table(embedding_table_size * embedding_dim);
    for (auto& val : embedding_table) {
        val = static_cast<float>(random_int(embedding_table_size));
    }

    // for (int i = 0; i < 10 && i < embedding_table.size(); ++i) {
    //     cout << embedding_table[i] << endl;
    // }
    

    // Input indices
    vector<int> input(input_size);
    for (auto& idx : input) {
        idx = random_int(embedding_table_size);
    }

    // Offsets
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) {
        offsets.push_back((input_size * i) / num_bags);
    }

    long long measured_us = 0;
    if (!enable_software_prefetch) {
        measured_us = naive_emb(embedding_table, input, offsets);
    } else {
        // Flush cache before prefetching run to avoid warm data bias
    for (size_t i = 0; i < embedding_table.size(); i += 16) {
        _mm_clflush(&embedding_table[i]);
    }
    _mm_mfence();
        measured_us = run_with_prefetching(embedding_table, input, offsets);
    }

    cout << "emb_exec_time_us: " << measured_us << "\n";
    return 0;
}