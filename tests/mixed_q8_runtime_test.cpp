// Exercise private projection dispatch with an independent dequantized oracle.
// Including the runtime TU avoids exposing its internal tensors as public API.
#include "../src/runtime/cpu_runtime.cpp"
#include <iostream>

int main() {
  using namespace qwen35x;
  constexpr std::size_t cols=128, main_rows=16, gates=32, blocks=cols/32;
  std::vector<float> source(main_rows*cols), gate_source(gates*cols);
  for (std::size_t i=0;i<source.size();++i) source[i]=std::sin(float(i)*.173F);
  for (std::size_t i=0;i<gate_source.size();++i) gate_source[i]=std::cos(float(i)*.071F);
  for (auto backend : {cpu::Q8_0Backend::scalar,cpu::Q8_0Backend::avx2,cpu::Q8_0Backend::auto_select}) {
    TensorData w;
    w.shape={main_rows+gates,cols}; w.q4_dot4=true;
    w.uses_q4_h128_transform=true; w.q4_h128_sign_seed=cpu::q4_h128_default_sign_seed;
    w.q8_0_backend=backend; w.q4_h128_signs.resize(1);
    cpu::q4_h128_prepare_signs(w.q4_h128_signs.data(),1,w.q4_h128_sign_seed);
    std::vector<cpu::Q4_0Block> canonical(main_rows*blocks);
    if (!cpu::q4_h128_quantize_matrix(source.data(),canonical.data(),main_rows,cols)) return 1;
    w.packed_q4_0_blocks.resize(main_rows/8*blocks);
    cpu::q4_dot4_pack_rows_8(canonical.data(),w.packed_q4_0_blocks.data(),main_rows,blocks);
    w.q8_gate_rows=gates; w.q8_gate_blocks.resize(gates*blocks);
    cpu::q8_0_quantize(gate_source.data(),w.q8_gate_blocks.data(),gates*blocks,cpu::Q8_0Backend::scalar);
    w.q8_gate_scales.resize(gates*blocks);
    cpu::q8_0_scales_to_f32(w.q8_gate_blocks.data(),w.q8_gate_scales.data(),gates*blocks);
    std::vector<float> q4_float(main_rows*cols),q8_float(gates*cols);
    cpu::q4_0_dequantize(canonical.data(),q4_float.data(),canonical.size(),cpu::Q8_0Backend::scalar);
    cpu::q8_0_dequantize(w.q8_gate_blocks.data(),q8_float.data(),gates*blocks,cpu::Q8_0Backend::scalar);
    for (int threads : {0,2}) {
      std::string error;
      auto context = threads ? make_cpu_execution_context(threads,error) : std::make_unique<CpuExecutionContext>();
      if (!context) return 2;
      for (std::size_t count : {1,2,3,4,5,16}) {
        std::vector<float> input(count*cols), output;
        for (std::size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*.113F)*3;
        if (!matmul_2d_quantized_batch(context.get(),w,input,count,output,error)) {
          std::cerr<<error; return 3;
        }
        if (output.size()!=count*(main_rows+gates)) return 4;
        for (std::size_t v=0;v<count;++v) {
          std::vector<cpu::Q8_0BlockX1> hinput(blocks);
          if (!prepare_q4_decode_activation(w,input.data()+v*cols,cols,hinput.data(),error)) return 5;
          std::vector<cpu::Q8_0Block> plain(blocks);
          std::vector<float> plain_float(cols);
          cpu::q8_0_quantize(input.data()+v*cols,plain.data(),blocks,backend);
          cpu::q8_0_dequantize(plain.data(),plain_float.data(),blocks,cpu::Q8_0Backend::scalar);
          for (std::size_t row=0;row<main_rows+gates;++row) {
            double expected=0, magnitude=0;
            for (std::size_t c=0;c<cols;++c) {
              const double product=row<main_rows
                ? double(q4_float[row*cols+c])*hinput[c/32].qs[c%32]*hinput[c/32].scales[0]
                : double(q8_float[(row-main_rows)*cols+c])*plain_float[c];
              expected+=product; magnitude+=std::abs(product);
            }
            if (std::abs(output[v*(main_rows+gates)+row]-expected)>1e-5*(1+magnitude)) {
              std::cerr<<"Mixed projection mismatch: "<<count<<" token "<<v<<" row "<<row; return 6;
            }
          }
          std::vector<float> one(input.begin()+v*cols,input.begin()+(v+1)*cols),single;
          if (!matvec_2d(context.get(),w,one,single,error)) return 7;
          for(std::size_t r=0;r<single.size();++r)
            if(std::abs(single[r]-output[v*single.size()+r])>1e-3F) return 8;
        }
      }
      // A Q8 head must use full logits for greedy selection, preserving penalties
      // and lowest-index ties rather than entering the packed-Q4-only shortcut.
      TensorData head;
      head.shape={gates,cols}; head.q8_0_backend=backend;
      head.q8_0_blocks=w.q8_gate_blocks; head.q8_0_scales=w.q8_gate_scales;
      std::vector<float> zero(cols,0),logits;
      std::vector<int> counts(gates,1);
      int selected=-1;
      if (!greedy_q4_token(context.get(),head,zero,counts,1.05F,selected,error) || selected!=0) return 9;
      for(std::size_t c=0;c<cols;++c) zero[c]=std::sin(float(c)*.23F);
      if (!matvec_2d(context.get(),head,zero,logits,error) ||
          !greedy_q4_token(context.get(),head,zero,counts,1.05F,selected,error)) return 10;
      int expected=0;
      for(std::size_t i=0;i<logits.size();++i) logits[i]=logits[i]>0?logits[i]/1.05F:logits[i]*1.05F;
      for(std::size_t i=1;i<logits.size();++i) if(logits[i]>logits[expected]) expected=static_cast<int>(i);
      if(selected!=expected) return 11;
    }
  }
  std::cout<<"Mixed H128-Q4/identity-Q8 rows match the FP64 oracle for scalar/SIMD, batches and tails\n";
}
