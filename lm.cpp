#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

//lm for to calibration
void lm(const mat &A, const vec p0)
{
    size_t cols = A.n_cols;
    size_t rows = A.n_rows;
    if (rows < cols)
    {
        cerr << "A should be overdetermint";
        abort();
    }
    if (cols != p0.n_rows)
    {
        cerr << "p is not as the same size as A.cols";
        abort();
    }
    // assert the size is suitable for all of A and p;

    mat At = A.t();
    mat pt = p0.t();

    mat AN = At * A; // for Jacobi J = A^t*A*p;
    mat f = pt * AN * p0;
    double epi = f(0, 0); // calculate the origin error;
    cout << "epi: " << epi;

    vec p = p0;

    mat res;
    res = A * p;
    res.print("res: ");

    p.print("before iteration: ");
    double lambda = 1;

    mat J = AN * p; // Jacobi matrix, also a vec;
    mat Jt = J.t();
    mat N = J * Jt;

    lambda = 1e-3 * mean(N.diag());

    N = N + (lambda * eye(AN.n_rows, AN.n_rows));

    bool first = false;
    bool second = false;

    int max_iteration = 60;
    while (max_iteration--)
    {

        mat Jep = J * (-epi);

        vec X;
        solve(X, N, Jep);
        vec p_temp = p + X;

        mat p_tempt = p_temp.t();
        f = p_tempt * AN * p_temp;

        double epi_temp = f(0, 0);
        // cout << "!";
        if(epi_temp < epi)
        {
            epi = epi_temp;
            p = p_temp;
            lambda /= 10;

            cout << " | ";
            if(first)
                second = true;
            if(!first)
                first = true;
                        
        }else
        {
            lambda *= 10;
            if(first && second)
                break;
            cout << " laeger " << endl;
        }
        // cout << "!";
        J = AN * p; // Jacobi matrix, also a vec;
        Jt = J.t();
        N = J * Jt + (lambda * eye(AN.n_rows, AN.n_rows));

    }

    p.print("after iteration: ");
    res = A * p;
    res.print("res: ");
}

void lm_tester()
{
    vec p(3);
    p = {20, 4, 15};
    p += 4 * randu<vec>(3);
    p.print("p: ");

    mat A(4, 3);
    A = {{0, 15, -4}, {-3, 0, 4}, {2, 5, -4}, {1, 10, -4}};
    A.print("A: ");

    cout << endl;
    lm(A, p);
}

int main(int argc, char const *argv[])
{
    lm_tester();
    return 0;
}
