#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>

// #define PRINT_RESULT_NEEDED

void print_matrix(float**, int);

void print_matrix(float** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << std::setw(9) << std::setprecision(3) << matrix[i][j] << std::setw(9);
        }
        std::cout << "\n";
    }
}

void swap_rows(float** matrix, int size, int row1, int row2) {
    for (int i = 0; i < size; ++i) {
        float tmp = matrix[row1][i];
        matrix[row1][i] = matrix[row2][i];
        matrix[row2][i] = tmp;
    }
}

void fill_matrix_as_unit(float** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (j == i) {
                matrix[i][i] = 1;
            } else {
                matrix[i][j] = 0;
            }
        }
    }
}

void lup_decompose(float** a, float** l, float** u, float** p, int size) {
    // fill P as unit matrix
    fill_matrix_as_unit(p, size);

    for (int k = 0; k < size; ++k) {
        float pivot = 0.0;
        int pivot_index = k;

        for (int i = k; i < size; ++i) {
            if (std::fabs(a[i][k]) > pivot) {
                pivot = std::fabs(a[i][k]);
                pivot_index = i;
            }
        }

        if (pivot <= std::numeric_limits<float>::epsilon()) {
            throw std::runtime_error("Matrix is singular to machine precision.");
        }

        swap_rows(p, size, k, pivot_index);
        swap_rows(a, size, k, pivot_index);

        for (int i = k + 1; i < size; ++i) {
            a[i][k] = a[i][k] / a[k][k];
            for (int j = k + 1; j < size; ++j) {
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            }
        }
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; j++) {
            if (j < i) {
                u[i][j] = 0;
            } else {
                u[i][j] = a[i][j];
            }
        }
        for (int j = 0; j < size; ++j) {
            if (j > i) {
                l[i][j] = 0;
            }
            else if (j == i) {
                l[i][j] = 1;
            } else {
                l[i][j] = a[i][j];
            }
        }
    }
}

void initialize_matrices(float** a, float** l, float** u, float** p, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] = new float[size];
        l[i] = new float[size];
        u[i] = new float[size];
        p[i] = new float[size];
    }
}

void fill_matrix(float** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cin >> matrix[i][j];
        }
    }
}

int main() {
    double runtime;
    int size;
    std::cin >> size;

    float** a = new float* [size];
    float** l = new float* [size];
    float** u = new float* [size];
    float** p = new float* [size];
    initialize_matrices(a, l, u, p, size);
    fill_matrix(a, size);

    runtime = clock()/(double)CLOCKS_PER_SEC;
    lup_decompose(a, l, u, p, size);
    runtime = (clock()/(double)CLOCKS_PER_SEC) - runtime;

#ifdef PRINT_RESULT_NEEDED
    std::cout << "L Matrix:\n";
    print_matrix(l, size);
    std::cout << "U Matrix:\n";
    print_matrix(u, size);
    std::cout << "P Matrix:\n";
    print_matrix(p, size);
#endif
    std::cout << "Runtime: " << runtime << " seconds\n";
    return 0;
}
