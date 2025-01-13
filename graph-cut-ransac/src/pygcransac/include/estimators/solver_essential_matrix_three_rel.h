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
// Relative depth solver from Fast Relative Pose Estimation using Relative Depth
// J. Astermark, Y. Ding, V. Larsson, A. Heyden, Fast Relative Pose Estimation using Relative Depth. 3DV 2024.

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
			class EssentialMatrixThreeRelSolver : public SolverEngine
			{
			public:
				EssentialMatrixThreeRelSolver()
				{
				}

				~EssentialMatrixThreeRelSolver()
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
					return "rel3";
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

			OLGA_INLINE bool EssentialMatrixThreeRelSolver::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				Eigen::MatrixXd x1(3, 3);
				Eigen::MatrixXd x2(3, 3);
				double sigma[3];

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					x1.col(i) << data_ptr[offset], data_ptr[offset + 1], 1.0;
					x2.col(i) << data_ptr[offset + 2], data_ptr[offset + 3], 1.0;
					sigma[i] = data_ptr[offset + 5] / (data_ptr[offset + 4]);
				}

				const double c0 = sigma[0] * sigma[0],
							 c1 = sigma[1] * sigma[1],
							 c2 = x1.col(0).squaredNorm(),
							 c3 = x1.col(1).squaredNorm(),
							 c4 = x2.col(0).squaredNorm() * c0,
							 c5 = x2.col(1).squaredNorm() * c1,
							 c6 = 2 * sigma[0] * sigma[1] * x2.col(0).transpose() * x2.col(1),
							 c7 = 2 * x1.col(0).transpose() * x1.col(1),
							 b0 = 2 * sigma[0] * x2.col(0).transpose() * x2.col(2),
							 b1 = 2 * x1.col(0).transpose() * x1.col(2),
							 b2 = 2 * sigma[1] * x2.col(1).transpose() * x2.col(2),
							 b3 = 2 * x1.col(1).transpose() * x1.col(2),
							 b4 = x1.col(2).squaredNorm(),
							 b5 = x2.col(2).squaredNorm();

				double lambda1[2];
				static const double TOL_IMAG = 1e-1;
				bool real1[2];
				poselib::sturm::solve_quadratic_real_tol(c5 - c3, c7 - c6, c4 - c2, lambda1, real1, TOL_IMAG);

				for (int k = 0; k < 2; ++k)
				{
					if (!real1[k])
						continue; // Imaginary solution
					if (lambda1[k] < 0)
						continue; // Negative depth

					const double a0 = b2 * lambda1[k] - b0, a1 = (b3 * lambda1[k] - b1) / a0,
								 a2 = (c2 - c4 + (c5 - c3) * lambda1[k] * lambda1[k]) / a0;

					double lambda2[2];
					bool real2[2];
					poselib::sturm::solve_quadratic_real_tol(b5 * a1 * a1 - b4, b1 - a1 * b0 + 2 * a1 * a2 * b5,
															 b5 * a2 * a2 - b0 * a2 - c2 + c4, lambda2, real2, TOL_IMAG);

					for (int m = 0; m < 2; ++m)
					{
						if (!real2[m])
							continue;
						if (lambda2[m] < 0)
							continue;

						double lambda2s = a1 * lambda2[m] + a2;
						if (lambda2s < 0)
							continue;

						Eigen::Vector3d v1 = sigma[0] * x2.col(0) - sigma[1] * lambda1[k] * x2.col(1);
						Eigen::Vector3d v2 = sigma[0] * x2.col(0) - lambda2s * x2.col(2);
						Eigen::Matrix3d Y;
						Y << v1, v2, v1.cross(v2);

						Eigen::Vector3d u1 = x1.col(0) - lambda1[k] * x1.col(1);
						Eigen::Vector3d u2 = x1.col(0) - lambda2[m] * x1.col(2);
						Eigen::Matrix3d X;
						X << u1, u2, u1.cross(u2);
						X = X.inverse().eval();

						Eigen::Matrix3d rot = Y * X;

						Eigen::Vector3d trans1 = rot * x1.col(0);
						Eigen::Vector3d trans2 = sigma[0] * x2.col(0);
						Eigen::Vector3d trans = trans2 - trans1;
						Eigen::Matrix3d TX;
						TX << 0, -trans(2), trans(1),
							trans(2), 0, -trans(0),
							-trans(1), trans(0), 0;

						Eigen::Matrix<double, 3, 3> Ess;
						Ess = TX * rot;

						Model model;
						model.descriptor = Ess;
						models_.push_back(model);
					}
				}

				return true;
			}
		}
	}
}