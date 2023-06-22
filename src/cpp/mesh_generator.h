#ifndef _MESH_GENERATOR_H
#define _MESH_GENERATOR_H

#include <Eigen/Eigen>

namespace fwi_ls {

Eigen::Array<int, Eigen::Dynamic, 4> build_connectivity_list(int nx, int ny);
    
}

#endif
