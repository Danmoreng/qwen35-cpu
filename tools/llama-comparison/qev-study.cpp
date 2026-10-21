// Resident qev evaluation; CUDA is confined to the separate offline teacher build.
#include "llama.h"
#include "ggml-backend.h"
#include "capture-inputs.h"
#include "../qev-study-common.h"
#include <memory>
int main(int argc,char **argv) try {
  std::string model_path,jobs_path,letters_path,output,captures;int threads=8,context=32768;bool ack=false;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];if(arg=="--wait-for-ack"){ack=true;continue;}
    if(++i>=argc)throw std::runtime_error("Missing value");std::string value=argv[i];
    if(arg=="--weights")model_path=value;
    else if(arg=="--jobs-file")jobs_path=value;
    else if(arg=="--letters-file")letters_path=value;
    else if(arg=="--output")output=value;
    else if(arg=="--capture-root")captures=value;
    else if(arg=="--threads")threads=std::stoi(value);
    else if(arg=="--max-context")context=std::stoi(value);
    else throw std::runtime_error("Unknown argument: "+arg);
  }
  auto jobs=qev_study::jobs(jobs_path);auto letters=qev_study::tokens(letters_path);
  if(model_path.empty()||output.empty()||std::filesystem::exists(output)||threads<1||context<1)
    throw std::runtime_error("Require weights, fresh output, positive threads/context");
  if(!captures.empty()&&!std::filesystem::create_directories(captures))throw std::runtime_error("Require fresh capture root");
  ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();
#ifdef QWEN35_CALIBRATION_CUDA
  auto gpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
  if(!gpu || std::string(ggml_backend_dev_name(gpu)).rfind("CUDA",0)!=0)
    throw std::runtime_error("CUDA teacher requires a CUDA device");
  mp.n_gpu_layers=999;ggml_backend_dev_t devices[]={gpu,nullptr};mp.devices=devices;
#else
  mp.n_gpu_layers=0;ggml_backend_dev_t devices[]={nullptr};mp.devices=devices;
#endif
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);
  if(!model)throw std::runtime_error("Model load failed");
  ProjectionCapture capture;capture.reservoir=true;
#ifdef QWEN35_CALIBRATION_CUDA
  capture.bulk_download=true;
#endif
  auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_seq_max=1;cp.n_batch=2048;cp.n_ubatch=512;
  cp.n_threads=threads;cp.n_threads_batch=threads;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;
  cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
#ifdef QWEN35_CALIBRATION_CUDA
  cp.offload_kqv=true;cp.op_offload=true;
#else
  cp.offload_kqv=false;cp.op_offload=false;
#endif
  if(!captures.empty()){cp.cb_eval=ProjectionCapture::callback;cp.cb_eval_user_data=&capture;}
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
  if(!ctx)throw std::runtime_error("Context creation failed");
  int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  llama_batch batch=llama_batch_init(2048,0,1);
  struct Guard{llama_batch &b;~Guard(){llama_batch_free(b);}}guard{batch};
  std::ofstream out(output);if(!out)throw std::runtime_error("Cannot create output");
  for(const auto &job:jobs) {
    auto tokens=qev_study::tokens(job.path);
    if(tokens.size()>std::size_t(context))throw std::runtime_error("Context exceeded: "+job.id);
    for(auto token:tokens)if(token<0||token>=vocab)throw std::runtime_error("Invalid input token");
    llama_synchronize(ctx.get());llama_memory_clear(llama_get_memory(ctx.get()),true);
    if(!captures.empty()) {capture.entries.clear();capture.error.clear();capture.directory=std::filesystem::path(captures)/job.id;
      if(!std::filesystem::create_directory(capture.directory))throw std::runtime_error("Duplicate capture job");}
    for(int pos=0;pos<int(tokens.size());) {
      int count=std::min(2048,int(tokens.size())-pos);batch.n_tokens=count;
      for(int i=0;i<count;++i){batch.token[i]=tokens[pos+i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=pos+i+1==int(tokens.size());}
      if(llama_decode(ctx.get(),batch)!=0)throw std::runtime_error("Decode failed: "+job.id);
      llama_synchronize(ctx.get());pos+=count;
    }
    if(!captures.empty())capture.finish();
    auto logits=llama_get_logits_ith(ctx.get(),-1);if(!logits)throw std::runtime_error("Missing logits");
    qev_study::write(out,job.id,logits,vocab,letters);
    std::cout<<"DONE\t"<<job.id<<std::endl;
    if(ack){std::string reply;if(!std::getline(std::cin,reply)||reply!=job.id)throw std::runtime_error("Missing capture acknowledgement");}
  }
  return 0;
} catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
