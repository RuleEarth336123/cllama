#include "base/base.h"
#include "base/tick.h"
#include "glog/logging.h"
#include "models/qwen2.h"

using std::string;
using std::vector;
using namespace tensor;

int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,bool need_output = false){
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL,tokens.empty()) << "the tokens is empty.";

    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    std::vector<int32_t> words;
    words.push_back(next);
    while(pos < total_steps){
        pos_tensor.index<int32_t>(0) = pos;
        if(pos < prompt_len - 1){
            Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }else{
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        if(model.is_sentence_ending(next)){
            break;
        }
        // string word;
        if(is_prompt){
            next = tokens.at(pos + 1);
            words.push_back(next);
            // word = model.decode(next);
        }else{
            words.push_back(next);
            // word = model.decode(next);
        }

        // if(need_output){
        //     std::cout << word.c_str() <<std::endl;
        // }

        pos += 1;
    }

    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    return std::min(pos,total_steps);
}

int main(int argc,char* argv[]){
    using namespace std;
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path ";
        return -1;
    }
    const char* checkpoint_path = argv[1]; 
    const char* tokenizer_path = argv[2];

    model::Qwen2Model model(base::TokenizerType::kEncodeBpe,tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCPU);
    if(!init_status){
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }
    cout<<"can i help you? tell me your question..."<<endl;
    const std::string& sentence = "你好";
    generate(model, sentence, 128, true);
    cout<<endl;
    cout<< "continue?" <<endl;

    // while(1){
    //     cin >> sentence;
    //     generate(model, sentence, 128, true);
    //     cout<<endl;
    //     cout<< "continue?" <<endl;
    // }
    // const std::string& sentence = "你好";

    return 0;
}