// Copyright (C) 2025, Yaqing Ding.
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
// Author: Yaqing Ding. Fundamental matrix solver from
// Y. Ding, V. Vávra, S. Bhayani, Q. Wu, J. Yang, Z. Kukelova, Fundamental Matrix Estimation Using Relative Depths, ECCV 2024
#pragma once

#include "estimators/solver_engine.h"
#include "estimators/fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrix4p4dSolver : public SolverEngine
			{
			public:
				FundamentalMatrix4p4dSolver()
				{
				}

				~FundamentalMatrix4p4dSolver()
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
					return "4p4d";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sampleNumber_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool FundamentalMatrix4p4dSolver::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				Eigen::MatrixXd coefficients(12, 12);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double c[4];

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					const double
						u11 = data_ptr[offset],
						v11 = data_ptr[offset + 1],
						u12 = data_ptr[offset + 2],
						v12 = data_ptr[offset + 3],
						q1 = data_ptr[offset + 4],
						q2 = data_ptr[offset + 5],
						o1 = data_ptr[offset + 6],
						o2 = data_ptr[offset + 7];

					const double
						q = q2 / q1;

					coefficients(row, 0) = -u11;
					coefficients(row, 1) = -v11;
					coefficients(row, 2) = -1;
					coefficients(row, 3) = 0;
					coefficients(row, 4) = 0;
					coefficients(row, 5) = 0;
					coefficients(row, 6) = 0;
					coefficients(row, 7) = 0;
					coefficients(row, 8) = 0;
					coefficients(row, 9) = 0;
					coefficients(row, 10) = q;
					coefficients(row, 11) = -q * v12;
					++row;

					coefficients(row, 0) = 0;
					coefficients(row, 1) = 0;
					coefficients(row, 2) = 0;
					coefficients(row, 3) = -u11;
					coefficients(row, 4) = -v11;
					coefficients(row, 5) = -1;
					coefficients(row, 6) = 0;
					coefficients(row, 7) = 0;
					coefficients(row, 8) = 0;
					coefficients(row, 9) = -q;
					coefficients(row, 10) = 0;
					coefficients(row, 11) = q * u12;
					++row;

					if (i == 3)
						break;

					coefficients(row, 0) = 0;
					coefficients(row, 1) = 0;
					coefficients(row, 2) = 0;
					coefficients(row, 3) = 0;
					coefficients(row, 4) = 0;
					coefficients(row, 5) = 0;
					coefficients(row, 6) = -u11;
					coefficients(row, 7) = -v11;
					coefficients(row, 8) = -1;
					coefficients(row, 9) = q * v12;
					coefficients(row, 10) = -q * u12;
					coefficients(row, 11) = 0;
					++row;
				}

				Eigen::Matrix<double, 12, 1> f1 = coefficients.block<11, 11>(0, 0).partialPivLu().solve(-coefficients.block<11, 1>(0, 11)).homogeneous();

				FundamentalMatrix model;
				model.descriptor << f1[0], f1[1], f1[2],
					f1[3], f1[4], f1[5],
					f1[6], f1[7], f1[8];

				models_.push_back(model);

				return models_.size();
			}
		}
	}
}