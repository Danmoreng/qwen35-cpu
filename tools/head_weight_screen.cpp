#include "qwen35x/weights/safetensors.h"
#include "qwen35x/cpu/q4_quantizer.h"
#include "qwen35x/cpu/q4_h128.h"
#include "../src/cpu/q8_0_internal.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Weight-only development screen. G16 uses the production MSE16 fitter on
// duplicated halves: this multiplies its objective by two without changing
// its minimizer. The parent G32 candidate remains an explicit fallback.
double error(const float* w, const qwen35x::cpu::Q4_0Block& q, int n=32) {
  double result=0;
  const double d=qwen35x::cpu::detail::half_to_float(q.d);
  for(int i=0;i<n;++i) {
    const int code=((q.qs[i%16]>>(i<16?0:4))&15)-8;
    const double delta=w[i]-d*code;
    result+=delta*delta;
  }
  return result;
}
int main(int argc,char** argv) try {
  if(argc!=4) throw std::runtime_error("Usage: head_weight_screen MODEL_DIR OUTPUT_CSV SAMPLE_ROWS");
  const auto count=std::stoull(argv[3]);
  qwen35x::SafetensorTensorF32 tensor;
  std::string failure;
  if(!qwen35x::SafetensorLoader::read_tensor_f32(argv[1],
      "model.language_model.embed_tokens.weight",tensor,failure)) throw std::runtime_error(failure);
  if(tensor.shape!=std::vector<std::int64_t>{248320,1024} || count==0 || count>248320)
    throw std::runtime_error("Expected Qwen3.5-0.8B tied matrix and 1..248320 rows");
  std::ofstream out(argv[2]);
  if(!out) throw std::runtime_error("Cannot open output");
  out<<"basis,seed,group,row,stratum,energy,sse\n"<<std::setprecision(17);
  const std::array<std::uint64_t,4> seeds={qwen35x::cpu::q4_h128_default_sign_seed,1,42,2026};
  for(int basis=0;basis<5;++basis) {
    const auto seed=basis?seeds[basis-1]:0;
    double total32=0,total16=0,energy=0;
    for(std::size_t sample=0;sample<count;++sample) {
      // Midpoints of equally sized vocabulary strata; frozen before evaluation.
      const auto row=((2*sample+1)*248320)/(2*count);
      const float* original=tensor.data.data()+row*1024;
      std::array<float,1024> transformed;
      std::copy_n(original,1024,transformed.data());
      if(basis && !qwen35x::cpu::q4_h128_transform_rows(original,transformed.data(),1,1024,seed))
        throw std::runtime_error("Transform failed");
      double e=0,s32=0,s16=0;
      for(int i=0;i<1024;++i) e+=double(original[i])*original[i];
      for(int group=0;group<32;++group) {
        const float* w=transformed.data()+32*group;
        qwen35x::cpu::Q4_0Block parent;
        if(!qwen35x::cpu::q4_quantize_offline(w,&parent,1,qwen35x::cpu::Q4Quantizer::mse16))
          throw std::runtime_error("Quantizer failed");
        const double parent_error=error(w,parent);
        double split_error=0;
        for(int half=0;half<2;++half) {
          float duplicate[32];
          for(int i=0;i<32;++i) duplicate[i]=w[half*16+i%16];
          qwen35x::cpu::Q4_0Block child;
          if(!qwen35x::cpu::q4_quantize_offline(duplicate,&child,1,qwen35x::cpu::Q4Quantizer::mse16))
            throw std::runtime_error("Half quantizer failed");
          double parent_half_error=0;
          const double d=qwen35x::cpu::detail::half_to_float(parent.d);
          for(int i=0;i<16;++i) {
            const int q=((parent.qs[i]>>(half*4))&15)-8;
            const double delta=w[half*16+i]-d*q;
            parent_half_error+=delta*delta;
          }
          split_error+=std::min(parent_half_error,error(duplicate,child,16));
        }
        s32+=parent_error;
        s16+=std::min(parent_error,split_error);
      }
      for(int group:{32,16}) out<<(basis?"h128":"identity")<<','<<seed<<','<<group<<','<<row<<','<<row*16/248320<<','<<e<<','<<(group==32?s32:s16)<<'\n';
      energy+=e; total32+=s32; total16+=s16;
    }
    std::cout<<(basis?"h128":"identity")<<" seed="<<seed<<" normalized_sse_g32="<<total32/energy<<" normalized_sse_g16="<<total16/energy<<std::endl;
  }
  if(!out) throw std::runtime_error("Output write failed");
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
