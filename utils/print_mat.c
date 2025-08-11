
void print_flat_2d_mat(int m, int n, float* A){
    for (int i =0; i<m;i++)
        for (int j =0; j<n; j++)
            printf("%.2f ", A[i * n + j]);
}
