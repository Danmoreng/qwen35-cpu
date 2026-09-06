#include "qwen35x/weights/gguf.h"
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace qwen35x;
void u32(std::ostream & f, std::uint32_t v) { for(int i=0;i<4;++i) f.put(char(v>>(8*i))); }
void u64(std::ostream & f, std::uint64_t v) { for(int i=0;i<8;++i) f.put(char(v>>(8*i))); }
void str(std::ostream & f, const std::string & s) { u64(f,s.size()); f.write(s.data(),s.size()); }
void text(std::ostream & f, const std::string & key, const std::string & value) {
  str(f,key);u32(f,8);str(f,value);
}
void number(std::ostream & f, const std::string & key, std::uint32_t value) {
  str(f,"qwen35."+key);u32(f,4);u32(f,value);
}
void real(std::ostream & f, const std::string & key, float value) {
  str(f,"qwen35."+key);u32(f,6);u32(f,std::bit_cast<std::uint32_t>(value));
}
int main() {
  const auto file = std::filesystem::temp_directory_path()/
    ("q35-profile-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");
  struct Cleanup { std::filesystem::path p; ~Cleanup(){std::error_code ec;std::filesystem::remove(p,ec);} } cleanup{file};
  ModelProfile p;
  auto & c=p.text;
  c.num_hidden_layers=24;c.hidden_size=1024;c.intermediate_size=3584;c.max_position_embeddings=262144;
  c.num_attention_heads=8;c.num_key_value_heads=2;c.head_dim=256;c.linear_conv_kernel_dim=4;
  c.linear_key_head_dim=128;c.linear_value_head_dim=128;c.linear_num_key_heads=16;c.linear_num_value_heads=16;
  c.full_attention_interval=4;c.vocab_size=248320;
  for(int i=0;i<24;++i)p.fingerprint.attention_schedule.push_back((i+1)%4?AttentionBlock::linear:AttentionBlock::full);
  for(bool nonfinite : {false,true}) {
    std::ofstream f(file,std::ios::binary);
    f.write("GGUF",4);u32(f,3);u64(f,0);u64(f,22);
    text(f,"general.architecture","qwen35");text(f,"tokenizer.ggml.model","gpt2");text(f,"tokenizer.ggml.pre","qwen35");
    number(f,"block_count",24);number(f,"embedding_length",1024);number(f,"feed_forward_length",3584);
    number(f,"context_length",262144);number(f,"attention.head_count",8);number(f,"attention.head_count_kv",2);
    number(f,"attention.key_length",256);number(f,"attention.value_length",256);
    real(f,"attention.layer_norm_rms_epsilon",nonfinite?std::numeric_limits<float>::quiet_NaN():c.rms_norm_eps);
    real(f,"rope.freq_base",c.rope_theta);number(f,"rope.dimension_count",64);
    number(f,"ssm.conv_kernel",4);number(f,"ssm.state_size",128);number(f,"ssm.group_count",16);
    number(f,"ssm.time_step_rank",16);number(f,"ssm.inner_size",2048);number(f,"full_attention_interval",4);
    str(f,"qwen35.rope.dimension_sections");u32(f,9);u32(f,5);u64(f,4);
    for(auto v:{11,11,10,0})u32(f,v);
    str(f,"tokenizer.ggml.tokens");u32(f,9);u32(f,8);u64(f,248320);
    for(int i=0;i<248320;++i)u64(f,0);
    while(static_cast<std::streamoff>(f.tellp())%32)f.put(0);
    f.close();
    GgufReader reader;std::string error;
    if(nonfinite) { if(reader.open(file.string(),error))return 1;continue; }
    if(!reader.open(file.string(),error) || !reader.validate_profile(p,error)) {
      std::cerr<<error<<'\n';return 2;
    }
    for(int change=0;change<7;++change) {
      auto wrong=p;
      if(change==0)wrong.text.hidden_size++;
      if(change==1)wrong.text.rms_norm_eps*=2;
      if(change==2)wrong.text.rope_theta*=2;
      if(change==3)wrong.text.vocab_size--;
      if(change==4)wrong.text.tie_word_embeddings=false;
      if(change==5)wrong.fingerprint.attention_schedule[0]=AttentionBlock::full;
      if(change==6)wrong.text.linear_value_head_dim++;
      if(reader.validate_profile(wrong,error) || error.empty())return 3;
    }
    reader.close();
    if(reader.validate_profile(p,error))return 4;
  }
  std::cout<<"GGUF semantic profile tests passed\n";
}
