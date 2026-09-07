// Offline teacher collection only; no GPU backend is added to the CPU engine.
#include "llama.h"
#include "ggml-backend.h"
#include "../llama-comparison/capture-inputs.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
struct Job { std::string id,path; std::vector<llama_token> tokens; };
int main(int argc,char** argv) try {
  std::string model_path,jobs_path,profile_path,output_root;
  int threads=8,context=8192;bool batched=false,acknowledge=false;
  for(int i=1;i<argc;++i) {
    const std::string arg=argv[i];
    const auto value=[&](){if(++i>=argc)throw std::runtime_error("Missing value");return std::string(argv[i]);};
    if(arg=="--cpu-gguf")model_path=value();
    else if(arg=="--jobs-file")jobs_path=value();
    else if(arg=="--profile-json")profile_path=value();
    else if(arg=="--output-root")output_root=value();
    else if(arg=="--cpu-threads")threads=std::stoi(value());
    else if(arg=="--max-context")context=std::stoi(value());
    else if(arg=="--batched-teacher")batched=true;
    else if(arg=="--wait-for-ack")acknowledge=true;
    else throw std::runtime_error("Unknown option: "+arg);
  }
  if(output_root=="@profile")output_root=profile_path+".captures";
  if(model_path.empty() || jobs_path.empty() || profile_path.empty() || output_root.empty() || threads<=0 || context<=128)
    throw std::runtime_error("Required: --cpu-gguf --jobs-file --profile-json --output-root");
  std::ifstream jobs_file(jobs_path);std::vector<Job> jobs;std::string line;
  while(std::getline(jobs_file,line)) {
    if(!line.empty() && line.back()=='\r')line.pop_back();
    if(line.empty())continue;
    const auto tab=line.find('\t');
    if(tab==std::string::npos)throw std::runtime_error("Jobs require id TAB token-path");
    Job job{line.substr(0,tab),line.substr(tab+1),{}};
    if(job.id.empty() || job.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
      throw std::runtime_error("Unsafe job id");
    std::ifstream tokens(job.path);std::string token;
    if(!tokens)throw std::runtime_error("Missing token file");
    while(std::getline(tokens,token,','))job.tokens.push_back(std::stoi(token));
    if(job.tokens.size()<=128 || job.tokens.size()>std::size_t(context))throw std::runtime_error("Invalid token count");
    jobs.push_back(std::move(job));
  }
  if(jobs.empty() || std::filesystem::exists(output_root))throw std::runtime_error("Require jobs and fresh output root");
  std::filesystem::create_directories(output_root);
  ggml_backend_load_all();llama_backend_init();
  const auto load_start=Clock::now();auto mp=llama_model_default_params();
#ifdef QWEN35_CALIBRATION_CUDA
  mp.n_gpu_layers=999;
  auto gpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
  if(!gpu || std::string(ggml_backend_dev_name(gpu)).rfind("CUDA",0)!=0)
    throw std::runtime_error("CUDA device required; CPU fallback is disabled for this collector");
  ggml_backend_dev_t gpu_devices[]={gpu,nullptr};mp.devices=gpu_devices;
#else
  mp.n_gpu_layers=0;ggml_backend_dev_t devices[]={nullptr};mp.devices=devices;
#endif
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);
  if(!model)throw std::runtime_error("Model load failed");
  ProjectionCapture capture;capture.reservoir=true;
#ifdef QWEN35_CALIBRATION_CUDA
  capture.bulk_download=true;
#endif
  auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_seq_max=1;
  cp.n_batch=2048;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;
  cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
  cp.cb_eval=ProjectionCapture::callback;cp.cb_eval_user_data=&capture;
#ifdef QWEN35_CALIBRATION_CUDA
  cp.offload_kqv=true;cp.op_offload=true;
#else
  cp.offload_kqv=false;cp.op_offload=false;
#endif
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
  if(!ctx)throw std::runtime_error("Context creation failed");
  const double load_ms=ms(load_start);const auto collect_start=Clock::now();
  const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  llama_batch batch=llama_batch_init(2048,0,1);
  struct Guard { llama_batch & batch; ~Guard(){llama_batch_free(batch);} } guard{batch};
  std::ofstream profile(profile_path);profile<<std::setprecision(12);
  profile<<"{\"quality_capture\":true,\"resident\":true,\"model_loads\":1,\"batched_teacher\":"<<(batched?"true":"false")
    <<",\"load_time_ms\":"<<load_ms<<",\"documents\":[";
  bool first=true;
  for(const auto& job:jobs) {
    for(auto token:job.tokens)if(token<0 || token>=vocab)throw std::runtime_error("Token out of range");
    llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);llama_synchronize(ctx.get());
    capture.entries.clear();capture.error.clear();capture.directory=std::filesystem::path(output_root)/job.id;
    if(!std::filesystem::create_directory(capture.directory))throw std::runtime_error("Duplicate job id");
    const auto start=Clock::now();const int n=static_cast<int>(job.tokens.size());
    const auto evaluate=[&](int position,int count,bool all_outputs) {
      batch.n_tokens=count;
      for(int i=0;i<count;++i) {
        batch.token[i]=job.tokens[position+i];batch.pos[i]=position+i;
        batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
        batch.logits[i]=all_outputs ? position+i>=n-129 : position+i==n-129;
        if(!batched && count==1)batch.logits[i]=true;
      }
      if(llama_decode(ctx.get(),batch)!=0)throw std::runtime_error("Decode failed");
      llama_synchronize(ctx.get());
    };
    if(batched) {
      for(int pos=0;pos<n-1;) {const int count=std::min(2048,n-1-pos);evaluate(pos,count,true);pos+=count;}
    } else {
      for(int pos=0;pos<n-128;) {const int count=std::min(2048,n-128-pos);evaluate(pos,count,false);pos+=count;}
      for(int pos=n-128;pos<n-1;++pos)evaluate(pos,1,false);
    }
    capture.finish();
    const auto* logits=llama_get_logits_ith(ctx.get(),-1);
    if(!logits)throw std::runtime_error("Missing logits");
    for(int i=0;i<vocab;++i)if(!std::isfinite(logits[i]))throw std::runtime_error("Nonfinite logits");
    std::ofstream final(capture.directory/"final-logits.f32",std::ios::binary);
    final.write(reinterpret_cast<const char*>(logits),vocab*sizeof(float));
    if(!final)throw std::runtime_error("Could not write logits");
    if(!first)profile<<',';first=false;
    profile<<"{\"id\":\""<<job.id<<"\",\"tokens\":"<<n<<",\"collection_ms\":"<<ms(start)<<'}';profile.flush();
    std::cout<<"CAPTURED\t"<<job.id<<std::endl;
    if(acknowledge) {std::string ack;if(!std::getline(std::cin,ack) || ack!=job.id)throw std::runtime_error("Missing compaction acknowledgement");}
  }
  profile<<"],\"document_count\":"<<jobs.size()<<",\"collection_time_ms\":"<<ms(collect_start)<<"}\n";
  if(!profile)throw std::runtime_error("Profile write failed");
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
