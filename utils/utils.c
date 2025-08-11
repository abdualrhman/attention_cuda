void write_binary(const char* filename, float* data, int size) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        exit(1);
    }
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}
