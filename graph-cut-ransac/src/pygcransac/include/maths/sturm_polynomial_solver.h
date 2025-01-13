// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef POSELIB_MISC_STURM_H_
#define POSELIB_MISC_STURM_H_
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Eigen>
#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace poselib {
namespace sturm {

// Constructs the quotients needed for evaluating the sturm sequence.
template <int N> void build_sturm_seq(const double *fvec, double *svec) {

    double f[3 * N];
    double *f1 = f;
    double *f2 = f1 + N + 1;
    double *f3 = f2 + N;

    std::copy(fvec, fvec + (2 * N + 1), f);

    for (int i = 0; i < N - 1; ++i) {
        const double q1 = f1[N - i] * f2[N - 1 - i];
        const double q0 = f1[N - 1 - i] * f2[N - 1 - i] - f1[N - i] * f2[N - 2 - i];

        f3[0] = f1[0] - q0 * f2[0];
        for (int j = 1; j < N - 1 - i; ++j) {
            f3[j] = f1[j] - q1 * f2[j - 1] - q0 * f2[j];
        }
        const double c = -std::abs(f3[N - 2 - i]);
        const double ci = 1.0 / c;
        for (int j = 0; j < N - 1 - i; ++j) {
            f3[j] = f3[j] * ci;
        }

        // juggle pointers (f1,f2,f3) -> (f2,f3,f1)
        double *tmp = f1;
        f1 = f2;
        f2 = f3;
        f3 = tmp;

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * N - 3] = f1[0];
    svec[3 * N - 2] = f1[1];
    svec[3 * N - 1] = f2[0];
}

// Evaluates polynomial using Horner's method.
// Assumes that f[N] = 1.0
template <int N> inline double polyval(const double *f, double x) {
    double fx = x + f[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        fx = x * fx + f[i];
    }
    return fx;
}

// Daniel Thul is responsible for this template-trickery :)
template <int D> inline unsigned int flag_negative(const double *const f) {
    return ((f[D] < 0) << D) | flag_negative<D - 1>(f);
}
template <> inline unsigned int flag_negative<0>(const double *const f) { return f[0] < 0; }
// Evaluates the sturm sequence and counts the number of sign changes
template <int N, typename std::enable_if<(N < 32), void>::type * = nullptr>
inline int signchanges(const double *svec, double x) {

    double f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    // In testing this turned out to be slightly faster compared to a naive loop
    unsigned int S = flag_negative<N>(f);

    return __builtin_popcount((S ^ (S >> 1)) & ~(0xFFFFFFFF << N));
}

template <int N, typename std::enable_if<(N >= 32), void>::type * = nullptr>
inline int signchanges(const double *svec, double x) {

    double f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    int count = 0;
    bool neg1 = f[0] < 0;
    for (int i = 0; i < N; ++i) {
        bool neg2 = f[i + 1] < 0;
        if (neg1 ^ neg2) {
            ++count;
        }
        neg1 = neg2;
    }
    return count;
}

// Computes the Cauchy bound on the real roots.
// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
template <int N> inline double get_bounds(const double *fvec) {
    double max = 0;
    for (int i = 0; i < N; ++i) {
        max = std::max(max, std::abs(fvec[i]));
    }
    return 1.0 + max;
}

// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
template <int N>
void ridders_method_newton(const double *fvec, double a, double b, double *roots, int &n_roots, double tol) {
    double fa = polyval<N>(fvec, a);
    double fb = polyval<N>(fvec, b);

    if (!((fa < 0) ^ (fb < 0)))
        return;

    const double tol_newton = 1e-3;

    for (int iter = 0; iter < 30; ++iter) {
        if (std::abs(a - b) < tol_newton) {
            break;
        }
        const double c = (a + b) * 0.5;
        const double fc = polyval<N>(fvec, c);
        const double s = std::sqrt(fc * fc - fa * fb);
        if (!s)
            break;
        const double d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
        const double fd = polyval<N>(fvec, d);

        if (fd >= 0 ? (fc < 0) : (fc > 0)) {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if (fd >= 0 ? (fa < 0) : (fa > 0)) {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    // We switch to Newton's method once we are close to the root
    double x = (a + b) * 0.5;

    double fx, fpx, dx;
    const double *fpvec = fvec + N + 1;
    for (int iter = 0; iter < 10; ++iter) {
        fx = polyval<N>(fvec, x);
        if (std::abs(fx) < tol) {
            break;
        }
        fpx = static_cast<double>(N) * polyval<N - 1>(fpvec, x);
        dx = fx / fpx;
        x = x - dx;
        if (std::abs(dx) < tol) {
            break;
        }
    }

    roots[n_roots++] = x;
}

template <int N>
void isolate_roots(const double *fvec, const double *svec, double a, double b, int sa, int sb, double *roots,
                   int &n_roots, double tol, int depth) {
    if (depth > 30)
        return;

    int n_rts = sa - sb;

    if (n_rts > 1) {
        double c = (a + b) * 0.5;
        int sc = signchanges<N>(svec, c);
        isolate_roots<N>(fvec, svec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
        isolate_roots<N>(fvec, svec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
    } else if (n_rts == 1) {
        ridders_method_newton<N>(fvec, a, b, roots, n_roots, tol);
    }
}

template <int N> inline int bisect_sturm(const double *coeffs, double *roots, double tol = 1e-10) {
    if (coeffs[N] == 0.0)
        return 0; // return bisect_sturm<N-1>(coeffs,roots,tol); // This explodes compile times...

    double fvec[2 * N + 1];
    double svec[3 * N];

    // fvec is the polynomial and its first derivative.
    std::copy(coeffs, coeffs + N + 1, fvec);

    // Normalize w.r.t. leading coeff
    double c_inv = 1.0 / fvec[N];
    for (int i = 0; i < N; ++i)
        fvec[i] *= c_inv;
    fvec[N] = 1.0;

    // Compute the derivative with normalized coefficients
    for (int i = 0; i < N - 1; ++i) {
        fvec[N + 1 + i] = fvec[i + 1] * ((i + 1) / static_cast<double>(N));
    }
    fvec[2 * N] = 1.0;

    // Compute sturm sequences
    build_sturm_seq<N>(fvec, svec);

    // All real roots are in the interval [-r0, r0]
    double r0 = get_bounds<N>(fvec);
    double a = -r0;
    double b = r0;

    int sa = signchanges<N>(svec, a);
    int sb = signchanges<N>(svec, b);

    int n_roots = sa - sb;
    if (n_roots == 0)
        return 0;

    n_roots = 0;
    isolate_roots<N>(fvec, svec, a, b, sa, sb, roots, n_roots, tol, 0);

    return n_roots;
}

template <> inline int bisect_sturm<1>(const double *coeffs, double *roots, double tol) {
    if (coeffs[1] == 0.0) {
        return 0;
    } else {
        roots[0] = -coeffs[0] / coeffs[1];
        return 1;
    }
}

template <> inline int bisect_sturm<0>(const double *coeffs, double *roots, double tol) { return 0; }


inline int solve_quadratic_real_tol(double a, double b, double c, double roots[2], bool real[2], double tol) {
    double b2 = b * b;
    double b2m4ac = b2 - 4 * a * c;
    if(b2m4ac > 0) {
        double sq = std::sqrt(b2m4ac);
        // Choose sign to avoid cancellations
        roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
        roots[1] = c / (a * roots[0]);
        real[0] = true;
        real[1] = true;
    } else {
        // We have some imaginary part
        double sq = std::sqrt(-b2m4ac);
        if(b > 0) {
            roots[0] = -(2*b*c)/(b2 - b2m4ac);
            real[0] = std::abs((2*c*sq)/(b2 - b2m4ac)) < tol;
            roots[1] = -b/(2*a);
            real[1] = std::abs(sq/(2*a)) < tol;
        } else {
            roots[0] = -(2*b*c)/(b2 - b2m4ac);
            real[0] = std::abs((2*c*sq)/(b2 - b2m4ac)) < tol;
            roots[1] = -b/(2*a);
            real[1] = std::abs(sq/(2*a)) < tol;
        }
    }
    return 2;
}

inline double sign(const double z) { return z < 0 ? -1.0 : 1.0; }

inline bool solve_cubic_single_real(double c2, double c1, double c0, double &root) {
    double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    if (c != 0) {
        if (c > 0) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
            return true;
        } else {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
    } else {
        root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
    }
    return false;
}

inline int solve_cubic_real(double c2, double c1, double c0, double roots[3]) {
    double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    int n_roots;
    if (c > 0) {
        c = std::sqrt(c);
        b *= -0.5;
        roots[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
        n_roots = 1;
    } else {
        c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        double d = 2.0 * std::sqrt(-a / 3.0);
        roots[0] = d * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        roots[1] = d * std::cos(std::acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0; // 2*pi/3
        roots[2] = d * std::cos(std::acos(c) / 3.0 - 4.18879020478639052527114472468384) - c2 / 3.0; // 4*pi/3
        n_roots = 3;
    }

    // single newton iteration
    for (int i = 0; i < n_roots; ++i) {
        double x = roots[i];
        double x2 = x * x;
        double x3 = x * x2;
        double dx = -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
        roots[i] += dx;
    }
    return n_roots;
}

inline int solve_quartic_real(double b, double c, double d, double e, double roots[4]) {

    // Find depressed quartic
    double p = c - 3.0 * b * b / 8.0;
    double q = b * b * b / 8.0 - 0.5 * b * c + d;
    double r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

    // Resolvent cubic is now
    // U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
    double bb = 2.0 * p;
    double cc = p * p - 4.0 * r;
    double dd = -q * q;

    // Solve resolvent cubic
    double u2;
    solve_cubic_single_real(bb, cc, dd, u2);

    if (u2 < 0)
        return 0;

    double u = sqrt(u2);

    double s = -u;
    double t = (p + u * u + q / u) / 2.0;
    double v = (p + u * u - q / u) / 2.0;

    int sols = 0;
    double disc = u * u - 4.0 * v;
    if (disc > 0) {
        roots[0] = (-u - sign(u) * std::sqrt(disc)) / 2.0;
        roots[1] = v / roots[0];
        sols += 2;
    }
    disc = s * s - 4.0 * t;
    if (disc > 0) {
        roots[sols] = (-s - sign(s) * std::sqrt(disc)) / 2.0;
        roots[sols + 1] = t / roots[sols];
        sols += 2;
    }

    for (int i = 0; i < sols; i++) {
        roots[i] = roots[i] - b / 4.0;

        // do one step of newton refinement
        double x = roots[i];
        double x2 = x * x;
        double x3 = x * x2;
        double dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
        roots[i] = x + dx;
    }
    return sols;
}

inline bool root2real(double b, double c, double &r1, double &r2) {
    double THRESHOLD = -1.0e-12;
    double v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    double y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

inline std::array<Eigen::Vector3d, 2> compute_pq(Eigen::Matrix3d C) {
    std::array<Eigen::Vector3d, 2> pq;
    Eigen::Matrix3d C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);

    pq[0] = C.col(0);
    pq[1] = C.row(0);

    return pq;
}

inline void refine_lambda(double &lambda1, double &lambda2, double &lambda3, const double a12, const double a13,
                          const double a23, const double b12, const double b13, const double b23) {

    for (int iter = 0; iter < 5; ++iter) {
        double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
            return;
        double x11 = lambda1 - lambda2 * b12;
        double x12 = lambda2 - lambda1 * b12;
        double x21 = lambda1 - lambda3 * b13;
        double x23 = lambda3 - lambda1 * b13;
        double x32 = lambda2 - lambda3 * b23;
        double x33 = lambda3 - lambda2 * b23;
        double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
        // This uses the closed form of the inverse for the jacobean.
        // Due to the zero elements this actually becomes quite nice.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

} // namespace sturm
} // namespace poselib

#endif