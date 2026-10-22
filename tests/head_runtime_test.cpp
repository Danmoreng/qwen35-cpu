#include "../src/runtime/cpu_runtime.cpp"
#include "qwen35x/cpu/q4_quantizer.h"
#include <iostream>
#include "attention_partition_cases.h"
int main() {
  using namespace qwen35x;
  if(!attention_partition_cases()) return 22;
  for(auto backend:{cpu::Q8_0Backend::scalar,cpu::Q8_0Backend::avx2,cpu::Q8_0Backend::auto_select})
    for(bool half:{false,true}) for(std::size_t count:{3U,33U}) for(int position:{0,37}) {
      constexpr std::size_t heads=8,kv_heads=2,dim=256,qw=heads*dim,kw=kv_heads*dim;
      const auto length=count+position;
      std::vector<float> q(count*qw),g(q.size()),k(length*kw),v(k.size()),a(q.size()),b(q.size()),scores(3*length);
      for(std::size_t i=0;i<q.size();++i){q[i]=std::sin(float(i)*.07F);g[i]=std::cos(float(i)*.13F);}
      for(std::size_t i=0;i<k.size();++i){k[i]=std::cos(float(i)*.09F);v[i]=std::sin(float(i)*.11F);}
      std::vector<std::uint16_t> kh(k.size()),vh(v.size());
      cpu::attention_cache_store_f16(k.data(),kh.data(),k.size(),backend);
      cpu::attention_cache_store_f16(v.data(),vh.data(),v.size(),backend);
      std::string error;auto context=make_cpu_execution_context(3,error);if(!context)return 24;
      FullAttentionBatchCpuJob job;
      job.queries=q.data();job.gates=g.data();job.k_cache=half?nullptr:k.data();job.v_cache=half?nullptr:v.data();
      job.k_cache_f16=half?kh.data():nullptr;job.v_cache_f16=half?vh.data():nullptr;
      job.scores=scores.data();job.output=a.data();job.context_stride=length;job.query_width=qw;job.kv_width=kw;
      job.position_start=position;job.head_count=heads;job.kv_head_count=kv_heads;job.head_dim=dim;
      job.attention_scale=1.F/16;job.backend=backend;job.score_partitions=3;job.attention_rows=count*heads;
      if(context->executor->parallel_for_rows(3,run_full_attention_batch_cpu_rows,&job)!=cpu::CpuExecutorStatus::ok)return 25;
      job.balance_rows=true;job.output=b.data();
      if(context->executor->parallel_for_rows(3,run_full_attention_batch_cpu_rows,&job)!=cpu::CpuExecutorStatus::ok ||
         std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))return 26;
    }
  constexpr std::size_t rows=264,cols=256,blocks=cols/32;
  for(bool rotated:{false,true}) for(bool g16:{false,true})
  for(auto backend:{cpu::Q8_0Backend::scalar,cpu::Q8_0Backend::avx2,cpu::Q8_0Backend::auto_select}) {
    TensorData w;w.shape={rows,cols};w.q8_0_backend=backend;
    w.uses_q4_h128_transform=rotated;w.q4_h128_sign_seed=42;
    std::vector<float> original(rows*cols),source(rows*cols);
    for(std::size_t i=0;i<source.size();++i) original[i]=std::sin(float(i)*.13F)*((i%37==0)?8:1);
    source=original;
    if(rotated && !cpu::q4_h128_transform_rows(source.data(),source.data(),rows,cols,42)) return 1;
    std::vector<cpu::Q4_0Block> parent(rows*blocks);
    if(!cpu::q4_quantize_offline(source.data(),parent.data(),parent.size(),cpu::Q4Quantizer::mse16)) return 2;
    if(g16) {
      w.g16_tiles.resize(rows/8*blocks);
      if(!cpu::q4_g16_pack(source.data(),w.g16_tiles.data(),rows,cols)) return 3;
    } else {
      w.q4_dot4=true;w.packed_q4_0_blocks.resize(rows/8*blocks);
      cpu::q4_dot4_pack_rows_8(parent.data(),w.packed_q4_0_blocks.data(),rows,blocks);
    }
    std::string error;auto ctx=make_cpu_execution_context(2,error);
    if(!ctx) return 4;
    std::vector<float> dequant(rows*cols),embedding(cols),parent_float(rows*cols);
    cpu::q4_0_dequantize(parent.data(),parent_float.data(),parent.size(),cpu::Q8_0Backend::scalar);
    for(std::size_t r=0;r<rows;++r) {
      if(g16) cpu::q4_g16_dequantize_row(w.g16_tiles.data(),r,dequant.data()+r*cols,blocks);
      else cpu::q4_dot4_dequantize_row(w.packed_q4_0_blocks.data(),r,dequant.data()+r*cols,blocks);
      dequantize_embedding_row(w,r,embedding.data(),cols);
      if(rotated && !cpu::q4_h128_transform_rows(embedding.data(),embedding.data(),1,cols,42)) return 5;
      double pe=0,ce=0;
      for(std::size_t c=0;c<cols;++c) {
        if(std::abs(embedding[c]-dequant[r*cols+c])>1e-5) return 6;
        pe+=std::pow(source[r*cols+c]-parent_float[r*cols+c],2);
        ce+=std::pow(source[r*cols+c]-dequant[r*cols+c],2);
      }
      if(g16 && ce>pe+1e-6) return 7;
    }
    for(std::size_t count:{1,2,3,4,5,8,15,16,17,31,32}) {
      std::vector<float> x(count*cols),out;
      for(std::size_t i=0;i<x.size();++i) x[i]=std::cos(float(i)*.19F);
      if(!matmul_2d_quantized_batch(ctx.get(),w,x,count,out,error)) {std::cerr<<error;return 8;}
      std::vector<float> merged;
      ctx->merge_tail_jobs=true;
      if(!matmul_2d_quantized_batch(ctx.get(),w,x,count,merged,error) ||
         merged.size()!=out.size() || std::memcmp(merged.data(),out.data(),out.size()*sizeof(float)))return 23;
      ctx->merge_tail_jobs=false;
      for(std::size_t t=0;t<count;++t) {
        std::vector<cpu::Q8_0BlockX1> a(blocks);
        if(rotated) {
          if(!cpu::q4_h128_prepare_activation_1(x.data()+t*cols,a.data(),cols,42,backend)) return 9;
        } else cpu::q8_0_quantize_vector_1(x.data()+t*cols,a.data(),blocks,backend);
        for(std::size_t r=0;r<rows;++r) {
          double expected=0,magnitude=0;
          for(std::size_t c=0;c<cols;++c) {
            const double product=double(dequant[r*cols+c])*a[c/32].qs[c%32]*a[c/32].scales[0];
            expected+=product;magnitude+=std::abs(product);
          }
          if(std::abs(out[t*rows+r]-expected)>1e-5*(1+magnitude)) {std::cerr<<"Head dot mismatch";return 10;}
        }
        std::vector<float> one(x.begin()+t*cols,x.begin()+(t+1)*cols),single;
        if(!matvec_2d(ctx.get(),w,one,single,error)) return 11;
        for(std::size_t r=0;r<rows;++r) if(std::abs(single[r]-out[t*rows+r])>1e-3) return 12;
        if(!g16) {
          const std::vector<std::int32_t> ids{263,0,8,7,8,17,262,1};std::vector<float> selected;
          if(!compute_selected_logits_from_embedding(ctx.get(),w,one,ids,selected,error))return 27;
          for(std::size_t i=0;i<ids.size();++i)
            if(std::memcmp(&selected[i],&single[ids[i]],sizeof(float)))return 28;
          const std::vector<std::int32_t> bad{-1};
          if(compute_selected_logits_from_embedding(ctx.get(),w,one,bad,selected,error))return 29;
        }
        std::vector<int> counts(rows,1);int token=-1;
        if(!greedy_q4_token(ctx.get(),w,one,counts,1.05F,token,error)) return 13;
        for(auto& v:single) v=v>0?v/1.05F:v*1.05F;
        if(token!=std::max_element(single.begin(),single.end())-single.begin()) return 14;
      }
    }
  }
  // Saturation boundary: independent signed half scales and full int8 inputs.
  for(auto byte:{0x00,0xff,0xf0,0x0f}) for(int activation:{-128,127}) {
    cpu::Q4G16Tile tile{};
    std::fill_n(tile.qs,128,static_cast<std::uint8_t>(byte));
    for(int r=0;r<8;++r) {tile.d[0][r]=0x3c00;tile.d[1][r]=0xbc00;}
    cpu::Q8_0BlockX1 a{};a.scales[0]=1;
    for(int i=0;i<32;++i) a.qs[i]=static_cast<std::int8_t>(i<16?activation:-activation-1);
    float scalar[8],simd[8],weights[32];
    cpu::q4_g16_matvec(&tile,&a,scalar,8,1,cpu::Q8_0Backend::scalar);
    cpu::q4_g16_matvec(&tile,&a,simd,8,1,cpu::Q8_0Backend::avx2);
    std::vector<cpu::Q8_0BlockX1> batch(17,a);
    std::vector<cpu::Q4G16ActivationTile> packed_batch(2);
    cpu::q4_g16_pack_activations(batch.data(),packed_batch.data(),17,1);
    for(auto backend:{cpu::Q8_0Backend::scalar,cpu::Q8_0Backend::avx2,cpu::Q8_0Backend::auto_select}) {
      float result[17*8];
      cpu::q4_g16_matmul(&tile,packed_batch.data(),result,8,17,1,8,backend);
      for(int i=0;i<17*8;++i) if(result[i]!=scalar[i%8]) return 16;
    }
    for(int r=0;r<8;++r) {
      cpu::q4_g16_dequantize_row(&tile,r,weights,1);
      double reference=0;
      for(int i=0;i<32;++i) reference+=weights[i]*a.qs[i];
      if(scalar[r]!=reference || simd[r]!=reference) return 15;
    }
  }
  // Weighted fitting keeps the parent's weighted objective representable in
  // each half. Unit importance must exactly reproduce the data-free fitter.
  std::vector<float> source(8*128),importance(128,1),dequant(128),parent_float(128);
  for(std::size_t i=0;i<source.size();++i) source[i]=std::sin(float(i)*.31F)*(i%9+1);
  std::vector<cpu::Q4G16Tile> plain(4),weighted(4);
  if(!cpu::q4_g16_pack(source.data(),plain.data(),8,128) ||
     !cpu::q4_g16_pack(source.data(),weighted.data(),8,128,importance.data()) ||
     std::memcmp(plain.data(),weighted.data(),4*sizeof(plain[0]))) return 17;
  for(int i=0;i<128;++i) importance[i]=i%3==0?.001F:float(i%17+1);
  if(!cpu::q4_g16_pack(source.data(),weighted.data(),8,128,importance.data())) return 18;
  for(int r=0;r<8;++r) {
    cpu::Q4_0Block parent[4];
    if(!cpu::q4_quantize_offline(source.data()+r*128,parent,4,cpu::Q4Quantizer::mse16,importance.data())) return 19;
    cpu::q4_0_dequantize(parent,parent_float.data(),4,cpu::Q8_0Backend::scalar);
    cpu::q4_g16_dequantize_row(weighted.data(),r,dequant.data(),4);
    double old_error=0,new_error=0;
    for(int c=0;c<128;++c) {
      old_error+=importance[c]*std::pow(source[r*128+c]-parent_float[c],2);
      new_error+=importance[c]*std::pow(source[r*128+c]-dequant[c],2);
    }
    if(new_error>old_error+1e-5) return 20;
  }
  importance[0]=-1;
  if(cpu::q4_g16_pack(source.data(),weighted.data(),8,128,importance.data())) return 21;
  std::cout<<"Head runtime tests passed\n";
}
