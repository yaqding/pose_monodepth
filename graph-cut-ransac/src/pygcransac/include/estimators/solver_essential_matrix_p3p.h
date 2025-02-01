// Copyright (C) 2025 Czech Technical University.
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
//
// Author: Yaqing Ding. P3P solver from
// Y. Ding, J. Yang, V. Larsson, C. Olsson, K. Åström, Revisiting the P3P Problem, CVPR 2023
#pragma once

#include "solver_engine.h"
#include "fundamental_estimator.h"
#include "../maths/sturm_polynomial_solver.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialMatrixP3PSolver : public SolverEngine
			{
			public:
				EssentialMatrixP3PSolver()
				{
				}

				~EssentialMatrixP3PSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				static constexpr const char *name()
				{
					return "P3PE";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 4;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sampleNumber_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool EssentialMatrixP3PSolver::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double sigma[3];

				Eigen::MatrixXd x1(3, 3);
				Eigen::MatrixXd x2(3, 3);

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					sigma[i] = data_ptr[offset + 4];

					x1.col(i) << sigma[i] * data_ptr[offset], sigma[i] * data_ptr[offset + 1], sigma[i];
					x2.col(i) << data_ptr[offset + 2], data_ptr[offset + 3], 1.0;
					x2.col(i).normalize();
				}

				Eigen::Vector3d X01 = x1.col(0) - x1.col(1);
				Eigen::Vector3d X02 = x1.col(0) - x1.col(2);
				Eigen::Vector3d X12 = x1.col(1) - x1.col(2);

				double a01 = X01.squaredNorm();
				double a02 = X02.squaredNorm();
				double a12 = X12.squaredNorm();

				std::array<Eigen::Vector3d, 3> X = {x1.col(0), x1.col(1), x1.col(2)};
				std::array<Eigen::Vector3d, 3> x = {x2.col(0), x2.col(1), x2.col(2)};

				if (a01 > a02)
				{
					if (a01 > a12)
					{
						std::swap(x[0], x[2]);
						std::swap(X[0], X[2]);
						std::swap(a01, a12);
						X01 = -X12;
						X02 = -X02;
					}
				}
				else if (a02 > a12)
				{
					std::swap(x[0], x[1]);
					std::swap(X[0], X[1]);
					std::swap(a02, a12);
					X01 = -X01;
					X02 = X12;
				}

				const double a12d = 1.0 / a12;
				const double a = a01 * a12d;
				const double b = a02 * a12d;

				const double m01 = x[0].dot(x[1]);
				const double m02 = x[0].dot(x[2]);
				const double m12 = x[1].dot(x[2]);

				// Ugly parameters to simplify the calculation
				const double m12sq = -m12 * m12 + 1.0;
				const double m02sq = -1.0 + m02 * m02;
				const double m01sq = -1.0 + m01 * m01;
				const double ab = a * b;
				const double bsq = b * b;
				const double asq = a * a;
				const double m013 = -2.0 + 2.0 * m01 * m02 * m12;
				const double bsqm12sq = bsq * m12sq;
				const double asqm12sq = asq * m12sq;
				const double abm12sq = 2.0 * ab * m12sq;

				const double k3_inv = 1.0 / (bsqm12sq + b * m02sq);
				const double k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
				const double k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
				const double k0 = k3_inv * (asqm12sq + a * m01sq);
				double s;
				bool G = poselib::sturm::solve_cubic_single_real(k2, k1, k0, s);

				Eigen::Matrix3d C;
				C(0, 0) = -a + s * (1 - b);
				C(0, 1) = -m02 * s;
				C(0, 2) = a * m12 + b * m12 * s;
				C(1, 0) = C(0, 1);
				C(1, 1) = s + 1;
				C(1, 2) = -m01;
				C(2, 0) = C(0, 2);
				C(2, 1) = C(1, 2);
				C(2, 2) = -a - b * s + 1;

				std::array<Eigen::Vector3d, 2> pq = poselib::sturm::compute_pq(C);

				double d0, d1, d2;

				Eigen::Matrix3d XX;

				XX << X01, X02, X01.cross(X02);
				XX = XX.inverse().eval();

				Eigen::Vector3d v1, v2;
				Eigen::Matrix3d YY;

				for (int i = 0; i < 2; ++i)
				{
					// [u1 u2 u3] * [lambda1; lambda2; lambda3] = 0
					double p0 = pq[i](0);
					double p1 = pq[i](1);
					double p2 = pq[i](2);
					// here we run into trouble if p0 is zero,
					// so depending on which is larger, we solve for either d0 or d1
					// The case p0 = p1 = 0 is degenerate and can be ignored
					bool switch_12 = std::abs(p0) <= std::abs(p1);

					if (switch_12)
					{
						// solve for lambda2
						double w0 = -p0 / p1;
						double w1 = -p2 / p1;
						double ca = 1.0 / (w1 * w1 - b);
						double cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
						double cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca;
						double taus[2];
						if (!poselib::sturm::root2real(cb, cc, taus[0], taus[1]))
							continue;
						for (double tau : taus)
						{
							if (tau <= 0)
								continue;
							// positive only
							d2 = std::sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
							d1 = tau * d2;
							d0 = (w0 * d2 + w1 * d1);
							if (d0 < 0)
								continue;

							poselib::sturm::refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
							v1 = d0 * x[0] - d1 * x[1];
							v2 = d0 * x[0] - d2 * x[2];
							YY << v1, v2, v1.cross(v2);
							Eigen::Matrix3d R = YY * XX;

							Eigen::Vector3d trans = d0 * x[0] - R * X[0];

							Eigen::Matrix3d TX;
							TX << 0, -trans(2), trans(1),
								trans(2), 0, -trans(0),
								-trans(1), trans(0), 0;

							Eigen::Matrix<double, 3, 3> Ess;
							Ess = TX * R;

							Model model;
							model.descriptor = Ess;
							models_.push_back(model);
						}
					}
					else
					{
						// Same as except we solve for lambda1 as a combination of lambda2 and lambda3
						// (default case in the paper)
						double w0 = -p1 / p0;
						double w1 = -p2 / p0;
						double ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1);
						double cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
						double cc = (1 - a * w0 * w0) * ca;

						double taus[2];
						if (!poselib::sturm::root2real(cb, cc, taus[0], taus[1]))
							continue;
						for (double tau : taus)
						{
							if (tau <= 0)
								continue;
							d0 = std::sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
							d1 = tau * d0;
							d2 = w0 * d0 + w1 * d1;

							if (d2 < 0)
								continue;

							poselib::sturm::refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
							v1 = d0 * x[0] - d1 * x[1];
							v2 = d0 * x[0] - d2 * x[2];
							YY << v1, v2, v1.cross(v2);
							Eigen::Matrix3d R = YY * XX;

							Eigen::Vector3d trans = d0 * x[0] - R * X[0];

							Eigen::Matrix3d TX;
							TX << 0, -trans(2), trans(1),
								trans(2), 0, -trans(0),
								-trans(1), trans(0), 0;

							Eigen::Matrix<double, 3, 3> Ess;
							Ess = TX * R;

							Model model;
							model.descriptor = Ess;
							models_.push_back(model);
						}
					}
				}

				return models_.size();
			}
		}
	}
}