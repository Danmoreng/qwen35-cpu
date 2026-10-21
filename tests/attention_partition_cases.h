#pragma once
#include "../src/runtime/attention_partition.h"
inline bool attention_partition_cases() {
  using namespace qwen35x::detail;
  for (std::size_t heads : {1U,2U,8U,16U})
    for (std::size_t tokens : {1U,2U,3U,7U,8U,31U,32U,33U,127U,128U,129U})
      for (std::size_t position : {0U,1U,31U,1024U,8192U})
        for (std::size_t requested : {1U,2U,4U,8U,16U}) {
          const auto rows = heads*tokens, parts = std::min(rows,requested);
          const auto total = attention_row_work(rows,heads,position);
          const auto maximum_row = position+tokens;
          std::size_t previous=0;
          std::uint64_t checked=0;
          for (std::size_t part=0;part<parts;++part) {
            const auto begin=attention_partition_boundary(part,parts,rows,heads,position);
            const auto end=attention_partition_boundary(part+1,parts,rows,heads,position);
            if(begin!=previous || end<=begin || end>rows)return false;
            std::uint64_t direct=0;
            for(auto row=begin;row<end;++row)direct+=position+row/heads+1;
            const auto cost=attention_row_work(end,heads,position)-attention_row_work(begin,heads,position);
            if(cost!=direct || cost>(total+parts-1)/parts+maximum_row)return false;
            checked+=direct;previous=end;
          }
          if(previous!=rows || checked!=total)return false;
        }
  return true;
}
