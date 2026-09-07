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
#include <cstdint>
#include <random>
#include <cstring>

struct ProjectionCapture {
  struct Entry { std::size_t columns=0, rows=0, output_rows=0, seen=0; std::mt19937_64 random{1234}; };
  std::filesystem::path directory;
  bool reservoir = false;
  bool bulk_download = false;
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
      std::vector<float> row(entry.columns);
      // A GPU download per retained row would serialize hundreds of tiny copies.
      std::vector<unsigned char> downloaded;
      if(self.bulk_download && x->ne[1]>0) {
        downloaded.resize((x->ne[1]-1)*x->nb[1]+row.size()*sizeof(float));
        ggml_backend_tensor_get(x,downloaded.data(),0,downloaded.size());
      }
      const auto path=self.directory/(name+".f32");
      if(!std::filesystem::exists(path)) { std::ofstream create(path,std::ios::binary); }
      std::fstream file(path,std::ios::binary|std::ios::in|std::ios::out);
      for(std::size_t i=0;i<std::size_t(x->ne[1]);++i) {
        const auto seen=entry.seen++;
        std::size_t slot=seen;
        if(seen>=256) {
          if(!self.reservoir)continue;
          slot=std::uniform_int_distribution<std::size_t>(0,seen)(entry.random);
          if(slot>=256)continue;
        }
        if(downloaded.empty())ggml_backend_tensor_get(x,row.data(),i*x->nb[1],row.size()*sizeof(float));
        else std::memcpy(row.data(),downloaded.data()+i*x->nb[1],row.size()*sizeof(float));
        for(float value:row)if(!std::isfinite(value))throw std::runtime_error("Non-finite captured input");
        file.seekp(slot*row.size()*sizeof(float));
        file.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));
        entry.rows=std::min(entry.seen,std::size_t(256));
      }
      if(!file)throw std::runtime_error("Could not write captured inputs");
    } catch(const std::exception & e) { self.error=e.what(); }
    return true;
  }
  void finish() {
    if(!error.empty())throw std::runtime_error(error);
    if(entries.empty())throw std::runtime_error("No projection inputs captured");
    std::ofstream file(directory/"capture.json");
    file<<"{\"version\":1,\"dtype\":\"little-endian-f32\",\"basis\":\"identity\",\"tensors\":[";
    file<<" ";
    bool first=true;
    for(const auto & [name,entry]:entries) {
      if(!first)file<<',';first=false;
      file<<"{\"name\":\""<<name<<"\",\"columns\":"<<entry.columns<<",\"samples\":"<<entry.rows
          <<",\"seen\":"<<entry.seen<<",\"output_rows\":"<<entry.output_rows<<'}';
    }
    file<<"]}\n";
    if(!file)throw std::runtime_error("Could not write capture manifest");
  }
};
