// Included after TensorData. Original packed bytes remain immutable and are
// shared by fused row views, including projections with mixed tensor types.
bool is_gguf_file(const std::string& path) {
  std::ifstream file(path,std::ios::binary);char magic[4]{};
  file.read(magic,4);return file && std::memcmp(magic,"GGUF",4)==0;
}
void dequantize_embedding_row(const TensorData& w,std::size_t row,
                              float* out,std::size_t cols) {
  if(!w.g16_tiles.empty()) {
    cpu::q4_g16_dequantize_row(w.g16_tiles.data(),row,out,cols/32);
    if(w.uses_q4_h128_transform) for(std::size_t b=0;b<cols/128;++b)
      cpu::q4_h128_inverse_block(out+b*128,b,w.q4_h128_sign_seed);
  } else if(!w.gguf_parts.empty()) {
    for(const auto& part:w.gguf_parts) {
      if(row<part.rows) {
        cpu::k_quant_dequantize(part.bytes->data()+row*(cols/256)*cpu::k_quant_block_bytes(part.type),
                                part.type,out,cols/256);return;
      }
      row-=part.rows;
    }
  } else if(w.is_q4_0()) {
    (w.q4_dot4?cpu::q4_dot4_dequantize_row:cpu::q4_0_packed_dequantize_row)(
      w.packed_q4_0_blocks.data(),row,out,cols/32);
    if(w.uses_q4_h128_transform)
      for(std::size_t block=0;block<cols/128;++block)
        cpu::q4_h128_inverse_block(out+block*128,block,w.q4_h128_sign_seed);
  } else {
    cpu::q8_0_dequantize(w.q8_0_blocks.data()+row*(cols/32),out,cols/32,w.q8_0_backend);
  }
}
struct GgufMatmulJob {
  const TensorData* weight;
  const cpu::KQuantActivation *k,*q8;
  float* output;
  std::size_t rows,blocks,count,row_tiles;
};
void gguf_matmul_rows(void* opaque,std::size_t begin,std::size_t end) noexcept {
  const auto& job=*static_cast<GgufMatmulJob*>(opaque);
  for(std::size_t task=begin;task<end;++task) {
    const std::size_t first_row=task%job.row_tiles*8;
    const std::size_t vector=task/job.row_tiles*4;
    for(std::size_t row=first_row;row<std::min(first_row+8,job.rows);++row) {
      std::size_t local=row;
      for(const auto& part:job.weight->gguf_parts) {
        if(local>=part.rows) {local-=part.rows;continue;}
        const auto* activation=part.type==cpu::KQuantType::q8_0?job.q8:job.k;
        cpu::k_quant_dot_rows(part.bytes->data()+local*job.blocks*cpu::k_quant_block_bytes(part.type),
          part.type,activation+vector*job.blocks,job.blocks,
          job.output+vector*job.rows+row,job.rows,std::min(std::size_t(4),job.count-vector),
          job.blocks,job.weight->q8_0_backend);
        break;
      }
    }
  }
}
bool gguf_matmul(CpuExecutionContext* context,const TensorData& w,
                 const std::vector<float>& input,std::size_t count,
                 std::vector<float>& output,std::string& error) {
  const std::size_t rows=w.shape[0],cols=w.shape[1],blocks=cols/256;
  if(!rows || !cols || cols%256 || count>std::numeric_limits<std::size_t>::max()/cols ||
     count>std::numeric_limits<std::size_t>::max()/rows || input.size()!=count*cols) {
    error="GGUF matmul input shape mismatch.";return false;
  }
  std::vector<cpu::KQuantActivation> local_k,local_q8;
  auto& k=context?context->k_input:local_k;
  auto& q8=context?context->k_q8_input:local_q8;
  bool need_k=false,need_q8=false;
  for(const auto& part:w.gguf_parts)
    (part.type==cpu::KQuantType::q8_0?need_q8:need_k)=true;
  CpuDecodeProbe probe(context,count==1?"k-quant-matvec":"k-quant-matmul",rows,cols,count);
  if(need_k) {k.resize(count*blocks);cpu::k_quant_prepare(input.data(),k.data(),count*blocks,false,w.q8_0_backend);}
  if(need_q8) {q8.resize(count*blocks);cpu::k_quant_prepare(input.data(),q8.data(),count*blocks,true,w.q8_0_backend);}
  output.resize(count*rows);probe.prepared();
  GgufMatmulJob job{&w,k.data(),q8.data(),output.data(),rows,blocks,count,(rows+7)/8};
  const auto tasks=job.row_tiles*((count+3)/4);
  if(context && context->executor) {
    if(context->executor->parallel_for_rows(tasks,gguf_matmul_rows,&job,probe.executor_timing())!=cpu::CpuExecutorStatus::ok) {
      error="GGUF matrix executor failed.";return false;
    }
  } else gguf_matmul_rows(&job,0,tasks);
  return true;
}

struct G16Job {
  const TensorData* w;const cpu::Q8_0BlockX1* a;float* out;
  const cpu::Q4G16ActivationTile* packed;
  std::size_t blocks,rows,tiles,count;
};
void g16_rows(void* opaque,std::size_t begin,std::size_t end) noexcept {
  const auto& j=*static_cast<G16Job*>(opaque);
  for(auto task=begin;task<end;) {
    const auto group=task/j.tiles,token=group*16,tile=(task%j.tiles)*32;
    const auto stop=std::min(end,(group+1)*j.tiles);
    const auto last=std::min(j.rows/8,(stop-group*j.tiles)*32);
    const auto tile_count=last-tile;
    if(j.count==1) {
      cpu::q4_g16_matvec(j.w->g16_tiles.data()+tile*j.blocks,j.a,
        j.out+tile*8,tile_count*8,j.blocks,j.w->q8_0_backend);
    } else cpu::q4_g16_matmul(j.w->g16_tiles.data()+tile*j.blocks,j.packed+(token/16)*j.blocks,
      j.out+token*j.rows+tile*8,tile_count*8,
      std::min(std::size_t(16),j.count-token),j.blocks,j.rows,j.w->q8_0_backend);
    task=stop;
  }
}
bool g16_matmul(CpuExecutionContext* ctx,const TensorData& w,const std::vector<float>& x,
    std::size_t count,std::vector<float>& out,std::string& error) {
  const std::size_t rows=w.shape[0],cols=w.shape[1],blocks=cols/32;
  if(!rows || rows%8 || !cols || cols%128 ||
      count>std::numeric_limits<std::size_t>::max()/cols ||
      count>std::numeric_limits<std::size_t>::max()/rows ||
      w.g16_tiles.size()!=(rows/8)*blocks || x.size()!=count*cols) {
    error="G16 matrix input mismatch";return false;
  }
  std::vector<cpu::Q8_0BlockX1> local;
  auto& prepared=ctx?ctx->prepared_q4_input:local;
  prepared.resize(count*blocks);
  CpuDecodeProbe probe(ctx,"head-g16-matmul",rows,cols,count);
  for(std::size_t t=0;t<count;++t) {
    if(w.uses_q4_h128_transform) {
      if(!cpu::q4_h128_prepare_activation_1(x.data()+t*cols,prepared.data()+t*blocks,
          cols,w.q4_h128_sign_seed,w.q8_0_backend)) {error="G16 activation transform failed";return false;}
    } else cpu::q8_0_quantize_vector_1(x.data()+t*cols,prepared.data()+t*blocks,blocks,w.q8_0_backend);
  }
  std::vector<cpu::Q4G16ActivationTile> local_packed;
  auto& packed=ctx?ctx->g16_prepared_batch:local_packed;
  if(count>1) {
    packed.resize(((count+15)/16)*blocks);
    cpu::q4_g16_pack_activations(prepared.data(),packed.data(),count,blocks);
  }
  out.resize(count*rows);probe.prepared();
  G16Job job{&w,prepared.data(),out.data(),packed.data(),blocks,rows,(rows/8+31)/32,count};
  const auto tasks=((count+15)/16)*job.tiles;
  if(ctx && ctx->executor) {
    if(ctx->executor->parallel_for_rows(tasks,g16_rows,&job,probe.executor_timing())!=cpu::CpuExecutorStatus::ok) {
      error="G16 executor failed";return false;
    }
  } else g16_rows(&job,0,tasks);
  return true;
}
