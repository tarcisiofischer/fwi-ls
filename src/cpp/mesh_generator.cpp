#include <mesh_generator.h>

Eigen::Array<int, Eigen::Dynamic, 4> fwi_ls::build_connectivity_list(int nx, int ny)
{
    auto from_ij_to_flat_index = [&nx](int i, int j) -> int {
        return (nx + 1) * i + j;
    };

    auto connectivity_list = Eigen::Array<int, Eigen::Dynamic, 4>(Eigen::Array<int, Eigen::Dynamic, 4>::Zero(nx * ny, 4));
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            auto idx = i * nx + j;

            connectivity_list(idx, 0) = from_ij_to_flat_index(i, j);
            connectivity_list(idx, 1) = from_ij_to_flat_index(i, j + 1);
            connectivity_list(idx, 2) = from_ij_to_flat_index(i + 1, j + 1);
            connectivity_list(idx, 3) = from_ij_to_flat_index(i + 1, j);
        }
    }
    return connectivity_list;
}
