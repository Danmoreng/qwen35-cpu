// Optional CPU teacher diagnostics. Never enable during speed measurements.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>

struct ProjectionCapture {
  struct Entry { std::size_t columns=0, rows=0, output_rows=0; };
  std::filesystem::path directory;
  std::map<std::string,Entry> entries;
  std::mutex mutex;
  std::string error;
  static bool callback(ggml_tensor * t, bool ask, void * opaque) noexcept {
    auto & self=*static_cast<ProjectionCapture *>(opaque);
    const auto * w=t->src[0]; const auto * x=t->src[1];
    if(t->op!=GGML_OP_MUL_MAT || !w || !x) return false;
    const std::string name=w->name;
    const bool wanted=(name.rfind("blk.",0)==0 || name=="output.weight" || name=="token_embd.weight") &&
                       name.size()>7 && name.substr(name.size()-7)==".weight";
    if(ask)return wanted;
    if(!wanted)return true;
    std::lock_guard<std::mutex> lock(self.mutex);
    try {
      if(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")!=std::string::npos)
        throw std::runtime_error("Unsafe captured tensor name");
      if(x->type!=GGML_TYPE_F32 || x->nb[0]!=sizeof(float) || x->ne[2]!=1 || x->ne[3]!=1 ||
         x->ne[0]!=w->ne[0] || (w->type!=GGML_TYPE_BF16 && w->type!=GGML_TYPE_F32))
        throw std::runtime_error("Capture requires BF16/F32 teacher weights and F32 matrix inputs: "+name);
      auto & entry=self.entries[name];
      if(entry.columns && entry.columns!=std::size_t(x->ne[0]))throw std::runtime_error("Capture shape changed");
      entry.columns=x->ne[0];entry.output_rows=w->ne[1];
      const std::size_t count=std::min(std::size_t(x->ne[1]),std::size_t(256)-entry.rows);
      if(!count)return true;
      std::vector<float> row(entry.columns);
      std::ofstream file(self.directory/(name+".f32"),std::ios::binary|std::ios::app);
      for(std::size_t i=0;i<count;++i) {
        ggml_backend_tensor_get(x,row.data(),i*x->nb[1],row.size()*sizeof(float));
        for(float value:row)if(!std::isfinite(value))throw std::runtime_error("Non-finite captured input");
        file.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));
      }
      if(!file)throw std::runtime_error("Could not write captured inputs");
      entry.rows+=count;
    } catch(const std::exception & e) { self.error=e.what(); }
    return true;
  }
  void finish() {
    if(!error.empty())throw std::runtime_error(error);
    if(entries.empty())throw std::runtime_error("No projection inputs captured");
    std::ofstream file(directory/"capture.json");
    file<<"{\"version\":1,\"dtype\":\"little-endian-f32\",\"basis\":\"identity\",\"tensors\":[";
    bool first=true;
    for(const auto & [name,entry]:entries) {
      if(!first)file<<',';first=false;
      file<<"{\"name\":\""<<name<<"\",\"columns\":"<<entry.columns<<",\"samples\":"<<entry.rows
          <<",\"output_rows\":"<<entry.output_rows<<'}';
    }
    file<<"]}\n";
    if(!file)throw std::runtime_error("Could not write capture manifest");
  }
};
