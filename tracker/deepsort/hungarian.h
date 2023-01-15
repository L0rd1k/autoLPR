#pragma once 

#include <vector>
#include <memory>
#include <cfloat>

namespace alpr {

class Hungarian {
public:
    double run(std::vector<std::vector<double>> &dist_matr, std::vector<int> &assignment) {
        if(dist_matr.empty()) {
            assignment.clear();
            return 0;
        }
        auto n_rows = dist_matr.size();
        auto n_cols = dist_matr[0].size();

        std::shared_ptr<double> dist_mat_in(new double[n_rows * n_cols], std::default_delete<double[]>());
        std::shared_ptr<int> assnment(new int[n_rows], std::default_delete<int[]>());
        auto cost = 0.0;
        for(uint i = 0; i < n_rows; i++) {
            for(uint j = 0; j < n_cols; j++) {
                dist_mat_in.get()[i + n_rows * j] = dist_matr[i][j];
            }
        }
        solve_optimal(assnment.get(), &cost, dist_mat_in.get(), n_rows, n_cols);
        assignment.clear();
        for(uint r = 0; r < n_rows; r++) {
            assignment.push_back(assnment.get()[r]);
        }
        return cost;
    }

    void solve_optimal(int *assnment, double *cost, double *dist_mat_in, int n_rows, int n_cols) {
        double *dist_mat;
        double *dist_mat_temp;
        double *dist_mat_end;
        double *column_end;
        double value, min_value;

        bool *covered_cols, *covered_rows;
        bool *star_mat;
        bool *new_star_mat;
        bool *prime_mat;

        int elements_n, min_dim, row, col;

        *cost = 0;
        for(row = 0; row < n_rows; row++) {
            assnment[row] = -1;
        }
        elements_n = n_rows * n_cols;
        dist_mat = (double*)malloc(elements_n * sizeof(double));
        dist_mat_end = dist_mat + elements_n;
        for(row = 0; row < elements_n; row++) {
            value = dist_mat_in[row];
            if( value < 0) {
                std::cerr << "All matrix elements have to be non-negative." << std::endl;
            }
            dist_mat[row] = value;
        }
        
        covered_cols = (bool*)calloc(n_cols, sizeof(bool));
        covered_rows = (bool*)calloc(n_rows, sizeof(bool));
        star_mat = (bool*)calloc(elements_n, sizeof(bool));
        prime_mat = (bool*)calloc(elements_n, sizeof(bool));
        new_star_mat = (bool*)calloc(elements_n, sizeof(bool));

        if(n_rows <= n_cols) {
            min_dim = n_rows;
            for(row = 0; row < n_rows; row++) {
                dist_mat_temp = dist_mat + row;
                min_value = *dist_mat_temp;
                dist_mat_temp += n_rows;
                while (dist_mat_temp < dist_mat_end) {
                    value = *dist_mat_temp;
                    if(value < min_value) {
                        min_value = value;
                    }
                    dist_mat_temp += n_rows;
                }

                dist_mat_temp = dist_mat + row;
                while (dist_mat_temp < dist_mat_end) {
                    *dist_mat_temp -= min_value;
                    dist_mat_temp += n_rows;
                }
            }

            for(row = 0; row < n_rows; row++) {
                for(col = 0; col < n_cols; col++) {
                    if(fabs(dist_mat[row + n_rows * col]) < DBL_EPSILON) {
                        if(!covered_cols[col]) {
                            star_mat[row + n_rows * col] = true;
                            covered_cols[col] = true;
                            break;
                        }
                    }
                }
            }
        } else {
            min_dim = n_cols;
            for(col = 0; col < n_cols; col++) {
                dist_mat_temp = dist_mat + n_rows * col;
                column_end = dist_mat_temp + n_rows;
                min_calue = *dist_mat_temp++;
                while (dist_mat_temp < column_end) {
                    value = *dist_mat_temp++;
                    if (value < min_value) {
                        min_value = value;
                    }
                }

                dist_mat_temp = dist_mat + n_rows * col;
                while(dist_mat_temp < column_end) {
                    *dist_mat_temp++ -= min_value;
                }
            }

            for(col = 0; col < n_cols; col++) {
                for(row = 0; row < n_rows; row++) {
                    if(fabs(dist_mat[row + n_rows * col]) < DBL_EPSILON) {
                        if(!covered_rows[row]) {
                            star_mat[row + n_rows * col] = true;
                            covered_cols[col] = true;
                            covered_rows[row] = true;
                            break;
                        }
                    }
                }
            }

            for(row = 0; row < n_rows; row++) {
                covered_rows[row] = false;
            }
        }

        


    }

};

}