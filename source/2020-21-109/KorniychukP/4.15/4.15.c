#include <stdio.h>
#include <stdlib.h>


void factorization(int number) {
    int p = 2, k = 1;

    while (number > 1) {
        if (number % p == 0) {
            number /= p;
            printf("%d", p);
            if (number % p == 0) {
                printf("^");
                do {
                    number /= p;
                    k++;
                } while (number % p == 0);
                printf("%d", k);
                k = 1;
            }
            if (number != 1) printf(" * ");
        }
        p += (1 + p % 2);
    }
}

int main() {
    FILE* input;
    int n;
    input = fopen("input.txt", "r");
    if (fscanf(input, "%d", &n) != 1) {
        printf("unable to read number\n");
        return -1;
    }
    if (n <= 1) {
        printf("incorrect value\n");
        return -1;
    }
    printf("%d = ", n);
    factorization(n);
    fclose(input);
    return 0;
}