// Real-model API contract checks; weights remain external to the repository.
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace qwen35x;
namespace {
struct Capture {
  std::vector<std::int32_t> ids;
  std::vector<float> expected;
  std::size_t vocab=0,calls=0;
};
bool dense(void *opaque,std::size_t index,std::int32_t token,const float *values,
           std::size_t count,std::string &error) {
  auto &c=*static_cast<Capture *>(opaque);
  if(index || token!=19 || count!=c.vocab){error="Full callback contract changed";return false;}
  c.expected.clear();for(auto id:c.ids)c.expected.push_back(values[id]);++c.calls;return true;
}
bool selected(void *opaque,std::size_t index,std::int32_t token,const float *values,
              std::size_t count,std::string &error) {
  auto &c=*static_cast<Capture *>(opaque);
  if(index || token!=19 || count!=c.expected.size() ||
     std::memcmp(values,c.expected.data(),count*sizeof(float))) {
    error="Selected callback differs from full head";return false;
  }
  ++c.calls;return true;
}
}
int main(int argc,char **argv)try {
  if(argc!=3)throw std::runtime_error("Usage: cpu_selected_logits_model_test MODEL_DIR Q4_ARTIFACT");
  std::string error;
  auto check=[&](bool ok){if(!ok)throw std::runtime_error("Selected API check failed: "+error);};
  auto profile=ProfileLoader::load_from_hf_directory(argv[1],error);check(bool(profile));
  CpuLoadOptions load;load.model_dir=argv[1];load.cpu_q4_h128_path=argv[2];
  auto model=CpuModel::load(*profile,load,error);check(bool(model));
  CpuEngineConfig cfg;cfg.threads=8;cfg.prefill_chunk_size=32;cfg.max_context=256;
  auto engine=CpuEngine::create(model,cfg,error);check(bool(engine));
  Capture capture;capture.vocab=profile->text.vocab_size;
  check(capture.vocab>1000);
  capture.ids={static_cast<std::int32_t>(capture.vocab-1),0,7,8,15,16,1000,8,0};
  auto run=[&](CpuRequestSpec spec) {
    capture.calls=0;auto id=engine->submit(std::move(spec),error);check(id!=0);
    CpuRequestResult result;
    do{check(engine->advance(id,error));check(engine->result(id,result));
       error=result.error;check(result.status!=CpuRequestStatus::failed);
    }while(result.status!=CpuRequestStatus::complete);
    check(capture.calls==1 && result.decode_forwards==0 && result.output_tokens==std::vector<std::int32_t>{19});
    check(engine->release(id));
  };
  auto pair=[&](CpuRequestSpec spec) {
    spec.logits_callback=dense;spec.logits_callback_context=&capture;run(spec);
    spec.logits_callback=nullptr;spec.logits_callback_context=nullptr;
    spec.selected_logit_tokens=capture.ids;spec.selected_logits_callback=selected;
    spec.selected_logits_callback_context=&capture;run(spec);return spec;
  };
  CpuRequestSpec last;
  for(std::size_t length:{1U,7U,31U,32U,33U,127U,128U,129U}) {
    CpuRequestSpec spec;for(std::size_t i=0;i<length;++i)spec.prompt_tokens.push_back(1+i%31);
    spec.max_new_tokens=1;spec.forced_output_tokens={19};spec.sampling.temperature=0;
    last=pair(spec);
  }
  for(std::size_t prefix_length:{31U,32U,33U,129U}) {
    auto spec=last;spec.selected_logit_tokens.clear();spec.selected_logits_callback=nullptr;
    spec.selected_logits_callback_context=nullptr;
    spec.trust_namespace="selected-head-test";
    spec.prefix=engine->register_prefix(std::span<const std::int32_t>(spec.prompt_tokens.data(),prefix_length),spec.trust_namespace,error);
    check(bool(spec.prefix));last=pair(spec);
  }
  auto reject=[&](CpuRequestSpec spec){check(engine->submit(std::move(spec),error)==0 && !error.empty());};
  auto bad=last;bad.selected_logit_tokens.clear();reject(bad);
  bad=last;bad.selected_logits_callback=nullptr;reject(bad);
  bad=last;bad.logits_callback=dense;reject(bad);
  bad=last;bad.max_new_tokens=2;bad.forced_output_tokens={19,19};reject(bad);
  bad=last;bad.forced_output_tokens.clear();reject(bad);
  bad=last;bad.selected_logit_tokens={-1};reject(bad);
  bad=last;bad.selected_logit_tokens={static_cast<std::int32_t>(capture.vocab)};reject(bad);
  run(last);
  std::cout<<"Selected API: dense/selected bits, duplicate and unordered IDs, vocabulary ends, 8 prompt lengths, 4 prefix hits and 7 invalid contracts passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
