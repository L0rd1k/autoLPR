#pragma once

#include <math.h>

#include <cfloat>
#include <memory>
#include <vector>

namespace alpr {

class Hungarian {
public:
    double run(std::vector<std::vector<double>> &dist_matr, std::vector<int> &assignment) {
        if (dist_matr.empty()) {
            assignment.clear();
            return 0;
        }
        auto n_rows = dist_matr.size();
        auto n_cols = dist_matr[0].size();

        std::shared_ptr<double> dist_mat_in(new double[n_rows * n_cols], std::default_delete<double[]>());
        std::shared_ptr<int> assnment(new int[n_rows], std::default_delete<int[]>());
        auto cost = 0.0;
        for (uint i = 0; i < n_rows; i++) {
            for (uint j = 0; j < n_cols; j++) {
                dist_mat_in.get()[i + n_rows * j] = dist_matr[i][j];
            }
        }
        solve_optimal(assnment.get(), &cost, dist_mat_in.get(), n_rows, n_cols);
        assignment.clear();
        for (uint r = 0; r < n_rows; r++) {
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
        for (row = 0; row < n_rows; row++) {
            assnment[row] = -1;
        }
        elements_n = n_rows * n_cols;
        dist_mat = (double *)malloc(elements_n * sizeof(double));
        dist_mat_end = dist_mat + elements_n;
        for (row = 0; row < elements_n; row++) {
            value = dist_mat_in[row];
            if (value < 0) {
                std::cerr << "All matrix elements have to be non-negative." << std::endl;
            }
            dist_mat[row] = value;
        }

        covered_cols = (bool *)calloc(n_cols, sizeof(bool));
        covered_rows = (bool *)calloc(n_rows, sizeof(bool));
        star_mat = (bool *)calloc(elements_n, sizeof(bool));
        prime_mat = (bool *)calloc(elements_n, sizeof(bool));
        new_star_mat = (bool *)calloc(elements_n, sizeof(bool));

        if (n_rows <= n_cols) {
            min_dim = n_rows;
            for (row = 0; row < n_rows; row++) {
                dist_mat_temp = dist_mat + row;
                min_value = *dist_mat_temp;
                dist_mat_temp += n_rows;
                while (dist_mat_temp < dist_mat_end) {
                    value = *dist_mat_temp;
                    if (value < min_value) {
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

            for (row = 0; row < n_rows; row++) {
                for (col = 0; col < n_cols; col++) {
                    if (fabs(dist_mat[row + n_rows * col]) < DBL_EPSILON) {
                        if (!covered_cols[col]) {
                            star_mat[row + n_rows * col] = true;
                            covered_cols[col] = true;
                            break;
                        }
                    }
                }
            }
        } else {
            min_dim = n_cols;
            for (col = 0; col < n_cols; col++) {
                dist_mat_temp = dist_mat + n_rows * col;
                column_end = dist_mat_temp + n_rows;
                min_value = *dist_mat_temp++;
                while (dist_mat_temp < column_end) {
                    value = *dist_mat_temp++;
                    if (value < min_value) {
                        min_value = value;
                    }
                }

                dist_mat_temp = dist_mat + n_rows * col;
                while (dist_mat_temp < column_end) {
                    *dist_mat_temp++ -= min_value;
                }
            }

            for (col = 0; col < n_cols; col++) {
                for (row = 0; row < n_rows; row++) {
                    if (fabs(dist_mat[row + n_rows * col]) < DBL_EPSILON) {
                        if (!covered_rows[row]) {
                            star_mat[row + n_rows * col] = true;
                            covered_cols[col] = true;
                            covered_rows[row] = true;
                            break;
                        }
                    }
                }
            }

            for (row = 0; row < n_rows; row++) {
                covered_rows[row] = false;
            }
        }

        step2b(assnment, dist_mat, star_mat, new_star_mat, prime_mat, covered_cols, covered_rows, n_rows, n_cols, min_dim);
        compute_assignement_cost(assnment, cost, dist_mat_in, n_rows);

        free(dist_mat);
        free(covered_cols);
        free(covered_rows);
        free(star_mat);
        free(prime_mat);
        free(new_star_mat);
        return;
    }

    void build_assignment_vec(int *assnment, bool *star_mat, int n_rows, int n_cols) {
        int row, col;
        for (row = 0; row < n_rows; row++) {
            for (col = 0; col < n_cols; col++) {
                if (star_mat[row + n_rows * col]) {
                    assnment[row] = col;
                    break;
                }
            }
        }
    }

    void compute_assignement_cost(int *assnment, double *cost, double *dist_mat, int n_rows) {
        int row, col;
        for (row = 0; row < n_rows; row++) {
            col = assnment[row];
            if (col >= 0) {
                *cost += dist_mat[row + n_rows * col];
            }
        }
    }

    void step2a(int *assnment, double *dist_mat, bool *star_mat,
                bool *new_star_mat, bool *prime_mat, bool *covered_cols,
                bool *covered_rows, int n_rows, int n_cols, int min_dim) {
        bool *star_mat_temp, *col_end;
        for (int col = 0; col < n_cols; col++) {
            star_mat_temp = star_mat + n_rows * col;
            col_end = star_mat_temp + n_rows;
            while (star_mat_temp < col_end) {
                if (*star_mat_temp++) {
                    covered_cols[col] = true;
                    break;
                }
            }
        }
        step2b(assnment, dist_mat, star_mat, new_star_mat, prime_mat,
               covered_cols, covered_rows, n_rows, n_cols, min_dim);
    }

    void step2b(int *assnment, double *dist_mat, bool *star_mat,
                bool *new_star_mat, bool *prime_mat, bool *covered_cols,
                bool *covered_rows, int n_rows, int n_cols, int min_dim) {
        int col, n_covered_cols = 0;
        for (col = 0; col < n_cols; col++) {
            if (covered_cols[col]) {
                n_covered_cols++;
            }
        }
        if (n_covered_cols = min_dim) {
            build_assignment_vec(assnment, star_mat, n_rows, n_cols);
        } else {
            step3(assnment, dist_mat, star_mat, new_star_mat, prime_mat, covered_cols, covered_rows, n_rows, n_cols, min_dim);
        }
    }

    void step3(int *assnment, double *dist_mat, bool *star_mat,
               bool *new_star_mat, bool *prime_mat, bool *covered_cols,
               bool *covered_rows, int n_rows, int n_cols, int min_dim) {
        bool zeros_found = true;
        int row, col, star_col;
        while (zeros_found) {
            zeros_found = false;
            for (col = 0; col < n_cols; col++) {
                if (!covered_cols[col]) {
                    for (row = 0; row < n_rows; row++) {
                        if ((!covered_rows[row]) && (fabs(dist_mat[row + n_rows * col]) < DBL_EPSILON)) {
                            prime_mat[row + n_rows * col] = true;
                            for (star_col = 0; star_col < n_cols; star_col++) {
                                if (star_mat[row + n_rows * star_col]) {
                                    break;
                                }
                            }
                            if (star_col == n_cols) {
                                step4(assnment, dist_mat, star_mat, new_star_mat, prime_mat,
                                      covered_cols, covered_rows, n_rows, n_cols, min_dim, row, col);
                                return;
                            } else {
                                covered_rows[row] = true;
                                covered_cols[star_col] = false;
                                zeros_found = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        step5(assnment, dist_mat, star_mat, new_star_mat, prime_mat, covered_cols, covered_rows, n_rows, n_cols, min_dim);
    }

    void step4(int *assnment, double *dist_mat, bool *star_mat,
               bool *new_star_mat, bool *prime_mat, bool *covered_cols,
               bool *covered_rows, int n_rows, int n_cols, int min_dim, int row, int col) {
        int n, star_row, star_col, prime_row, prime_col;
        int n_elements = n_rows * n_cols;
        for (n = 0; n < n_elements; n++) {
            new_star_mat[n] = star_mat[n];
        }
        new_star_mat[row + n_rows * col] = true;
        star_col = col;
        for (star_row = 0; star_row < n_rows; star_row++) {
            if (star_mat[star_row + n_rows * star_col]) {
                break;
            }
        }

        while (star_row < n_rows) {
            new_star_mat[star_row + n_rows * star_col] = false;
            prime_row = star_row;
            for (prime_col = 0; prime_col < n_cols; prime_col++) {
                if (prime_mat[prime_row + n_rows * prime_col]) {
                    break;
                }
            }
            new_star_mat[prime_row + n_rows * prime_col] = true;
            star_col = prime_col;
            for (star_row = 0; star_row < n_rows; star_row++) {
                if (star_mat[star_row + n_rows * star_col]) {
                    break;
                }
            }
        }

        for (n = 0; n < n_elements; n++) {
            prime_mat[n] = false;
            star_mat[n] = new_star_mat[n];
        }

        for (n = 0; n < n_rows; n++) {
            covered_rows[n] = false;
        }

        step2a(assnment, dist_mat, star_mat, new_star_mat, prime_mat,
               covered_cols, covered_rows, n_rows, n_cols, min_dim);
    }

    void step5(int *assnment, double *dist_mat, bool *star_mat,
               bool *new_star_mat, bool *prime_mat, bool *covered_cols,
               bool *covered_rows, int n_rows, int n_cols, int min_dim) {
        double value, h = DBL_MAX;
        int row, col;
        for (row = 0; row < n_rows; row++) {
            if (!covered_rows[row]) {
                for (col = 0; col < n_cols; col++) {
                    if (!covered_cols[col]) {
                        value = dist_mat[row + n_rows * col];
                        if (value < h) {
                            h = value;
                        }
                    }
                }
            }
        }
        for (row = 0; row < n_rows; row++) {
            if (covered_rows[row]) {
                for (col = 0; col < n_cols; col++) {
                    dist_mat[row + n_rows * col] += h;
                }
            }
        }
        for (col = 0; col < n_cols; col++) {
            if (!covered_cols[col]) {
                for (row = 0; row < n_rows; row++) {
                    dist_mat[row + n_rows * col] -= h;
                }
            }
        }
        step3(assnment, dist_mat, star_mat, new_star_mat, prime_mat,
              covered_cols, covered_rows, n_rows, n_cols, min_dim);
    }
};

}  // namespace alpr