// Offline numerical attribution on captured teacher inputs, never a speed test.
#include "qwen35x/weights/safetensors.h"
#include "qwen35x/weights/q4_h128_artifact.h"
#include "qwen35x/cpu/q4_dot4.h"
#include "qwen35x/cpu/q4_h128.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qwen35x;
int main(int argc,char ** argv) try {
  if(argc!=9)throw std::runtime_error("Expected HF-dir artifact source-tensor packed-tensor inputs.f32 source-row packed-row result.json");
  std::string error;
  SafetensorTensorF32 source;
  if(!SafetensorLoader::read_tensor_f32(argv[1],argv[3],source,error))throw std::runtime_error(error);
  Q4H128ArtifactReader reader;
  if(!reader.open(argv[2],error))throw std::runtime_error(error);
  const auto * info=reader.find_tensor(argv[4]);
  const auto source_row=std::stoull(argv[6]),packed_row=std::stoull(argv[7]);
  if(!info || source.shape.size()!=2 || info->shape.size()!=2 || !q4_h128_encoding_dot4(info->encoding) ||
      info->shape[1]!=static_cast<std::uint64_t>(source.shape[1]) || packed_row%8 ||
      packed_row>info->shape[0] || info->shape[0]-packed_row<8 ||
      source_row>static_cast<std::uint64_t>(source.shape[0]) || static_cast<std::uint64_t>(source.shape[0])-source_row<8)
    throw std::runtime_error("Invalid selected projection tile");
  const std::size_t columns=info->shape[1],blocks=columns/32;
  std::vector<std::uint8_t> bytes;
  if(!reader.read_tensor_bytes(argv[4],bytes,error))throw std::runtime_error(error);
  const auto * tile=reinterpret_cast<const cpu::Q4_0BlockX8*>(bytes.data())+(packed_row/8)*blocks;
  std::vector<float> weights(8*columns),x(columns),rotated(columns);
  for(int r=0;r<8;++r)cpu::q4_dot4_dequantize_row(tile,r,weights.data()+r*columns,blocks);
  std::vector<cpu::Q8_0BlockX1> prepared(blocks);
  std::ifstream inputs(argv[5],std::ios::binary);
  if(!inputs)throw std::runtime_error("Cannot read captured inputs");
  double squared[3]{},denominator=0,max_kernel=0;
  std::size_t samples=0;
  for(;samples<16;++samples) {
    inputs.read(reinterpret_cast<char*>(x.data()),columns*sizeof(float));
    if(!inputs)break;
    for(float v:x)if(!std::isfinite(v))throw std::runtime_error("Non-finite input");
    if(q4_h128_encoding_transformed(info->encoding)) {
      if(!cpu::q4_h128_transform_rows(x.data(),rotated.data(),1,columns,info->sign_seed) ||
         !cpu::q4_h128_prepare_activation_1(x.data(),prepared.data(),columns,info->sign_seed))
        throw std::runtime_error("Activation preparation failed");
    } else {
      rotated=x;cpu::q8_0_quantize_vector_1(x.data(),prepared.data(),blocks,cpu::Q8_0Backend::auto_select);
    }
    float kernel[8];cpu::q4_dot4_matvec(tile,prepared.data(),kernel,8,blocks);
    for(int r=0;r<8;++r) {
      double original=0,weight_only=0,actual_activation=0;
      for(std::size_t c=0;c<columns;++c) {
        original+=double(source.data[(source_row+r)*columns+c])*x[c];
        weight_only+=double(weights[r*columns+c])*rotated[c];
        actual_activation+=double(weights[r*columns+c])*prepared[c/32].scales[0]*prepared[c/32].qs[c%32];
      }
      const double delta[]={weight_only-original,actual_activation-weight_only,kernel[r]-actual_activation};
      for(int stage=0;stage<3;++stage)squared[stage]+=delta[stage]*delta[stage];
      denominator+=original*original;max_kernel=std::max(max_kernel,std::abs(delta[2]));
    }
  }
  if(!samples || denominator==0)throw std::runtime_error("Empty/zero-output probe");
  std::ofstream out(argv[8]);
  out<<std::setprecision(12)<<"{\"samples\":"<<samples<<",\"rows\":8,\"columns\":"<<columns
     <<",\"weight_rmse\":"<<std::sqrt(squared[0]/(samples*8))
     <<",\"activation_rmse\":"<<std::sqrt(squared[1]/(samples*8))
     <<",\"kernel_rmse\":"<<std::sqrt(squared[2]/(samples*8))
     <<",\"weight_relative_l2\":"<<std::sqrt(squared[0]/denominator)
     <<",\"activation_relative_l2\":"<<std::sqrt(squared[1]/denominator)
     <<",\"kernel_max_abs\":"<<max_kernel<<"}\n";
  if(!out)throw std::runtime_error("Cannot write attribution");
  return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<'\n';return 1; }
