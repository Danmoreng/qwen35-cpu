// Resident native evaluator; each job owns fresh recurrent/KV state, without prefix reuse.
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qev-study-common.h"
using namespace qwen35x;
struct Capture {
  std::vector<float> logits;
  static bool record(void *opaque, std::size_t, std::int32_t, const float *values, std::size_t n, std::string &) {
    static_cast<Capture *>(opaque)->logits.assign(values, values + n); return true;
  }
};
int main(int argc, char **argv) try {
  CpuLoadOptions load; CpuEngineConfig config; config.max_resident_requests=1;
  std::string jobs_path, letters_path, output;
  for (int i=1;i<argc;++i) {
    std::string arg=argv[i]; if (++i>=argc) throw std::runtime_error("Missing value"); std::string value=argv[i];
    if(arg=="--model-dir") load.model_dir=value;
    else if(arg=="--weights") load.cpu_q4_h128_path=value;
    else if(arg=="--jobs-file") jobs_path=value;
    else if(arg=="--letters-file") letters_path=value;
    else if(arg=="--output") output=value;
    else if(arg=="--threads") config.threads=std::stoi(value);
    else if(arg=="--max-context") config.max_context=std::stoul(value);
    else throw std::runtime_error("Unknown argument: "+arg);
  }
  auto jobs=qev_study::jobs(jobs_path); auto letters=qev_study::tokens(letters_path);
  if(output.empty() || std::filesystem::exists(output) || config.threads<1) throw std::runtime_error("Require fresh output and positive threads");
  std::string error; auto check=[&](bool ok){if(!ok)throw std::runtime_error(error);};
  auto profile=ProfileLoader::load_from_hf_directory(load.model_dir,error);check(bool(profile));
  auto model=CpuModel::load(*profile,load,error);check(bool(model));
  auto engine=CpuEngine::create(model,config,error);check(bool(engine));
  std::ofstream out(output); if(!out)throw std::runtime_error("Cannot create output");
  for(const auto &job:jobs) {
    Capture capture; CpuRequestSpec request; request.prompt_tokens=qev_study::tokens(job.path);
    request.max_new_tokens=1;request.forced_output_tokens={letters.front()};
    request.logits_callback=Capture::record;request.logits_callback_context=&capture;
    auto id=engine->submit(std::move(request),error);check(id!=0);CpuRequestResult result;
    for(;;) {check(engine->result(id,result));if(result.status==CpuRequestStatus::complete)break;
      if(result.status==CpuRequestStatus::failed)throw std::runtime_error(result.error);
      check(engine->advance(id,error));}
    if(capture.logits.size()!=std::size_t(profile->text.vocab_size))throw std::runtime_error("Missing full logits");
    qev_study::write(out,job.id,capture.logits.data(),capture.logits.size(),letters);
    check(engine->release(id));std::cout<<"DONE\t"<<job.id<<std::endl;
  }
  return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
