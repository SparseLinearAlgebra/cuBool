#pragma once

#include <grammar.h>
#include <response.h>

#ifdef __cplusplus
extern "C" {
#endif

int nsparse_cfpq(const Grammar *grammar, CfpqResponse *response,
         const GrB_Matrix *relations, const char **relations_names,
         size_t relations_count, size_t graph_size);

int nsparse_cfpq_index(const Grammar *grammar, CfpqResponse *response,
                 const GrB_Matrix *relations, const char **relations_names,
                 size_t relations_count, size_t graph_size);

#ifdef __cplusplus
}
#endif
