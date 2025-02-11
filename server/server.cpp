#include "base/base.h"
#include "base/tick.h"
#include "glog/logging.h"
#include "models/qwen2.h"
#include "httplib.h"

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
        string word;
        if(is_prompt){
            next = tokens.at(pos + 1);
            // words.push_back(next);
            word = model.decode(next);
        }else{
            // words.push_back(next);
            word = model.decode(next);
        }

        if(need_output){
            std::cout << word.c_str()<<" ";
        }

        pos += 1;
    }

    // if (need_output) {
    //     printf("%s ", model.decode(words).data());
    //     fflush(stdout);
    // }
    return std::min(pos,total_steps);
}
/*
    引入HTTP库（示例使用cpp-httplib）
    创建HTTP服务器实例
    定义POST接口/generate接收问题参数
    重定向cout输出到内存缓冲区以捕获生成结果
    持续监听HTTP请求而不是单次交互
*/

int main(int argc, char* argv[]) {
    using namespace std;
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path";
        return -1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DeviceType::kDeviceCPU);
    if (!init_status) {
        LOG(FATAL) << "Model init failed, error code: " << init_status.get_err_code();
    }

    httplib::Server server;
    
    // 定义RESTful接口
    server.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("question")) {
            res.status = 400;
            res.set_content("Missing 'question' parameter", "text/plain");
            return;
        }
        
        auto question = req.get_param_value("question");
        stringstream buffer;
        streambuf* old = cout.rdbuf(buffer.rdbuf());
        
        generate(model, question, 128, true);
        
        cout.rdbuf(old);
        res.set_content(buffer.str(), "text/plain");
    });

    cout << "Server running on http://localhost:8080" << endl;
    server.listen("0.0.0.0", 8080);

    return 0;
}