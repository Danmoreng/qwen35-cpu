#include "qwen35x/cpu/k_quant.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace qwen35x::cpu;
template<class T> std::vector<T> read(const std::filesystem::path& path) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file)throw std::runtime_error("Missing fixture");
  const auto bytes=static_cast<std::size_t>(file.tellg());
  std::vector<T> out(bytes/sizeof(T));file.seekg(0);
  file.read(reinterpret_cast<char*>(out.data()),bytes);
  if(!file)throw std::runtime_error("Truncated fixture");return out;
}
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv) try {
  check(argc==2,"Expected fixture directory");
  const char* names[]={"q4_k","q5_k","q6_k","q8_0"};
  const KQuantType types[]={KQuantType::q4_k,KQuantType::q5_k,KQuantType::q6_k,KQuantType::q8_0};
  for(int t=0;t<4;++t) {
    auto weights=read<std::uint8_t>(std::filesystem::path(argv[1])/(std::string(names[t])+".bin"));
    auto expected=read<float>(std::filesystem::path(argv[1])/(std::string(names[t])+".f32"));
    check(expected.size()==512 && weights.size()==2*k_quant_block_bytes(types[t]),"Fixture shape");
    std::vector<float> dequant(512);
    k_quant_dequantize(weights.data(),types[t],dequant.data(),2);
    for(std::size_t i=0;i<512;++i)check(dequant[i]==expected[i],"Independent GGML dequantization mismatch");
    for(auto backend:{Q8_0Backend::scalar,Q8_0Backend::avx2,Q8_0Backend::auto_select}) {
      // Include all-zero activations, both signs, nonuniform groups and ties.
      std::vector<float> x(4*512);
      for(std::size_t i=512;i<x.size();++i)x[i]=std::sin(float(i)*.17f)*(i%7+1);
      std::vector<KQuantActivation> activation(4*3);
      for(int v=0;v<4;++v)k_quant_prepare(x.data()+v*512,activation.data()+v*3,2,t==3,backend);
      for(int count=1;count<=4;++count) {
        float out[16];std::fill_n(out,16,123456.0f);
        k_quant_dot_rows(weights.data(),types[t],activation.data(),3,out,3,count,2,backend);
        for(int v=0;v<count;++v) {
          double reference=0,abs_sum=0;
          for(std::size_t i=0;i<512;++i) {
            const auto& a=activation[v*3+i/256];
            const double term=double(expected[i])*a.d[(i%256)/32]*a.qs[i%256];
            reference+=term;abs_sum+=std::abs(term);
          }
          check(std::abs(out[v*3]-reference)<1e-4+abs_sum*2e-6,"Quantized SIMD/reference dot mismatch");
          float single;
          k_quant_dot_rows(weights.data(),types[t],activation.data()+v*3,3,&single,1,1,2,backend);
          check(single==out[v*3],"Batch and single-row kernel mismatch");
        }
        for(int i=0;i<16;++i)if(i%3 || i/3>=count)check(out[i]==123456.0f,"Output stride overwrite");
      }
    }
  }
  KQuantActivation a{};float x[256]{};x[1]=4;
  k_quant_prepare(x,&a,1,false,Q8_0Backend::scalar);
  check(a.qs[1]==-127 && a.d[0]<0,"Signed Q8_K scale");
  x[1]=-4;k_quant_prepare(x,&a,1,false,Q8_0Backend::scalar);
  check(a.qs[1]==-127 && a.d[0]>0,"Negative Q8_K maximum");
  std::cout<<"K-quant independent fixtures, scalar/SIMD, zero, sign and stride tests passed\n";
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
