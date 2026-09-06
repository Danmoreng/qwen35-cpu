// Included after TensorData. Original packed bytes remain immutable and are
// shared by fused row views, including projections with mixed tensor types.
bool is_gguf_file(const std::string& path) {
  std::ifstream file(path,std::ios::binary);char magic[4]{};
  file.read(magic,4);return file && std::memcmp(magic,"GGUF",4)==0;
}
void dequantize_embedding_row(const TensorData& w,std::size_t row,
                              float* out,std::size_t cols) {
  if(!w.gguf_parts.empty()) {
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
