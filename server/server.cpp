#include "base/base.h"
#include "base/tick.h"
#include "glog/logging.h"
#include "models/qwen2.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using std::string;
using std::vector;
using namespace tensor;

int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps, bool need_output = false) {
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
        // 解析 JSON 请求体
        json request_json;
        try {
            request_json = json::parse(req.body);
        } catch (json::parse_error& e) {
            res.status = 400;
            res.set_content("Invalid JSON format", "text/plain");
            return;
        }

        // 检查必需的参数
        if (!request_json.contains("prompt")) {
            res.status = 400;
            res.set_content("Missing 'prompt' parameter", "text/plain");
            return;
        }

        string prompt = request_json["prompt"];
        int max_tokens = request_json.value("max_tokens", 100); // 默认值为100
        bool need_output = request_json.value("stream", false); // 是否流式输出

        stringstream buffer;
        streambuf* old = cout.rdbuf(buffer.rdbuf());

        // 生成文本
        int32_t generated_length = generate(model, prompt, max_tokens, need_output);

        cout.rdbuf(old);

        // 构建响应
        json response_json;
        response_json["id"] = "cmpl-" + std::to_string(rand()); // 生成一个随机ID
        response_json["object"] = "text_completion";
        response_json["created"] = static_cast<int32_t>(time(nullptr));
        response_json["model"] = "your_model_id"; // 替换为实际模型ID

        json choice;
        choice["text"] = buffer.str(); // 生成的文本
        choice["index"] = 0;
        choice["logprobs"] = nullptr; // 根据需要设置
        choice["finish_reason"] = "stop"; // 根据需要设置
        response_json["choices"] = { choice };

        // 计算 token 使用情况
        response_json["usage"]["prompt_tokens"] = prompt.size(); // 输入 token 数
        response_json["usage"]["completion_tokens"] = generated_length; // 生成 token 数
        response_json["usage"]["total_tokens"] = prompt.size() + generated_length; // 总 token 数

        // 设置响应内容
        res.set_content(response_json.dump(), "application/json");
    });

    cout << "Server running on http://localhost:8080" << endl;
    server.listen("0.0.0.0", 8080);

    return 0;
}