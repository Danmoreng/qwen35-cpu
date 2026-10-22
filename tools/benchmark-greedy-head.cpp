// Local diagnostic: true unforced greedy decoding, separated from Decision API.
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/cpu/q8_0.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace qwen35x;
using Json=nlohmann::ordered_json;
namespace {
bool full_callback(void *,std::size_t,std::int32_t,const float *values,
                   std::size_t count,std::string &error) {
  for(std::size_t i=0;i<count;++i)if(!std::isfinite(values[i])) {
    error="Non-finite full-head reference";return false;
  }
  return true;
}
}
int main(int argc,char **argv)try {
  std::string dir,weights,prompt_file,profile_file,mode="legacy";
  bool verify=false;
  for(int i=1;i<argc;i+=2) {
    if(i+1>=argc)throw std::runtime_error("Missing option value");
    const std::string key=argv[i],value=argv[i+1];
    if(key=="--model-dir")dir=value;
    else if(key=="--weights")weights=value;
    else if(key=="--prompt-tokens-file")prompt_file=value;
    else if(key=="--profile-json")profile_file=value;
    else if(key=="--mode")mode=value;
    else if(key=="--verify")verify=value=="true";
    else throw std::runtime_error("Unknown option: "+key);
  }
  if(dir.empty() || weights.empty() || prompt_file.empty() ||
     (mode!="legacy" && mode!="t8"))throw std::runtime_error("Invalid diagnostic arguments");
  std::string error;
  auto check=[&](bool ok){if(!ok)throw std::runtime_error("Greedy head check: "+error);};
  auto p=ProfileLoader::load_from_hf_directory(dir,error);check(bool(p));
  CpuLoadOptions load;load.model_dir=dir;load.cpu_q4_h128_path=weights;
  auto model=CpuModel::load(*p,load,error);check(bool(model));
  std::ifstream input(prompt_file);Json tokens;input>>tokens;
  const auto prompt=tokens.get<std::vector<std::int32_t>>();
  constexpr std::size_t steps=16;
  auto run=[&](std::size_t count,const std::string &kind) {
    CpuEngineConfig cfg;cfg.threads=8;cfg.prefill_chunk_size=32;cfg.max_context=256;
    cfg.max_prefix_bytes=0;cfg.batch_greedy=kind!="full";cfg.greedy_head_t8=kind=="t8";
    auto engine=CpuEngine::create(model,cfg,error);check(bool(engine));
    std::vector<CpuRequestId> ids;
    for(std::size_t r=0;r<count;++r) {
      CpuRequestSpec spec;spec.prompt_tokens=prompt;
      spec.prompt_tokens.push_back(static_cast<std::int32_t>(19+r));
      spec.prompt_tokens.push_back(198);
      spec.max_new_tokens=steps+1;spec.sampling.temperature=0;
      spec.sampling.repetition_penalty=1.05F+0.01F*static_cast<float>(r);
      if(kind=="full")spec.logits_callback=full_callback;
      const auto id=engine->submit(std::move(spec),error);check(id!=0);ids.push_back(id);
      CpuRequestResult result;
      do {check(engine->advance(id,error));check(engine->result(id,result));
          if(result.status==CpuRequestStatus::failed)error=result.error;
          check(result.status!=CpuRequestStatus::failed);
      }while(result.status==CpuRequestStatus::prefill);
      check(result.status==CpuRequestStatus::decode && result.output_tokens.size()==1);
    }
    const auto before=engine->stats();
    const auto start=std::chrono::steady_clock::now();
    for(std::size_t i=0;i<steps;++i)check(engine->advance_batch(ids,error));
    const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const auto after=engine->stats();Json output=Json::array();
    for(auto id:ids) {
      CpuRequestResult result;check(engine->result(id,result));
      if(result.status==CpuRequestStatus::failed)error=result.error;
      check(result.status==CpuRequestStatus::complete && result.decode_forwards==steps &&
            result.output_tokens.size()==steps+1);
      output.push_back(result.output_tokens);check(engine->release(id));
    }
    const auto t8=after.greedy_t8_head_batches-before.greedy_t8_head_batches;
    const bool supported=cpu::q8_0_resolve_backend(cpu::Q8_0Backend::auto_select)==cpu::Q8_0Backend::avx512_vnni;
    check(t8==((kind=="t8" && supported && count%16>=8)?steps:0));
    return Json{{"batch_size",count},{"mode",kind},{"decode_steps",steps},
      {"decode_ms",ms},{"batch_step_ms",ms/steps},{"aggregate_tokens_per_second",count*steps*1000.0/ms},
      {"greedy_head_batches",after.greedy_lm_head_batches-before.greedy_lm_head_batches},
      {"t8_head_batches",t8},{"output_tokens",output}};
  };
  Json report={{"engine","native-unforced-greedy-diagnostic"},{"threads",8},{"chunk",32},
    {"backend",cpu::q8_0_backend_name(cpu::q8_0_resolve_backend(cpu::Q8_0Backend::auto_select))},
    {"prefill_timed",false},{"session_warmup",true},{"results",Json::array()}};
  if(verify) {
    for(std::size_t b:{1U,4U,8U,12U,16U}) {
      auto full=run(b,"full"),legacy=run(b,"legacy"),t8=run(b,"t8");
      error="Generated token mismatch at B="+std::to_string(b);
      check(full["output_tokens"]==legacy["output_tokens"] && full["output_tokens"]==t8["output_tokens"]);
      report["results"].push_back({{"batch_size",b},{"tokens_equal",true},{"t8_batches",t8["t8_head_batches"]}});
      std::cout<<"B="<<b<<": unforced full/legacy/T8 tokens match; T8 batches="<<t8["t8_head_batches"]<<'\n';
    }
    report["validation"]="passed";
  } else {
    for(std::size_t b:{1U,4U,8U,12U,16U})run(b,mode);
    for(std::size_t b:{1U,4U,8U,12U,16U})report["results"].push_back(run(b,mode));
  }
  if(!profile_file.empty()){std::ofstream out(profile_file);check(bool(out));out<<report.dump(2)<<'\n';}
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
