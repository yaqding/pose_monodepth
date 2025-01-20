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
// Author: Yaqing Ding. Varying focal lengths solver from
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
			class FundamentalMatrixPT4D3 : public SolverEngine
			{
			public:
				FundamentalMatrixPT4D3()
				{
				}

				~FundamentalMatrixPT4D3()
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
					return "4p3d";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 3;
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

			OLGA_INLINE bool FundamentalMatrixPT4D3::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				Eigen::MatrixXd coefficients(10, 12);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;

				size_t row = 0;
				for (int i = 0; i < 4; i++)
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

					if (i < 3)
					{
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
					else
					{
						coefficients(row, 0) = u12 * u11;
						coefficients(row, 1) = u12 * v11;
						coefficients(row, 2) = u12;
						coefficients(row, 3) = v12 * u11;
						coefficients(row, 4) = v12 * v11;
						coefficients(row, 5) = v12;
						coefficients(row, 6) = u11;
						coefficients(row, 7) = v11;
						coefficients(row, 8) = 1;
						coefficients(row, 9) = 0;
						coefficients(row, 10) = 0;
						coefficients(row, 11) = 0;
						++row;
					}
				}

				Eigen::Matrix<double, 12, 1> n1, n2;
				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
				if (lu.dimensionOfKernel() != 2)
					return false;

				const Eigen::Matrix<double, 12, 2> null_space =
					lu.kernel();

				n1 = null_space.col(0);
				n2 = null_space.col(1);

				n1 = n1 / n1(11);
				n2 = n2 / n2(11);
				n1 = n1 - n2;
				n1 = n1 / n1(10);
				n2 = n2 - n1 * n2(10);

				double n11 = n1(0);
				double n12 = n1(1);
				double n13 = n1(2);
				double n14 = n1(3);
				double n15 = n1(4);
				double n16 = n1(5);
				double n17 = n1(6);
				double n18 = n1(7);
				double n19 = n1(8);
				double m11 = n1(9);

				double n21 = n2(0);
				double n22 = n2(1);
				double n23 = n2(2);
				double n24 = n2(3);
				double n25 = n2(4);
				double n26 = n2(5);
				double n27 = n2(6);
				double n28 = n2(7);
				double n29 = n2(8);
				double m21 = n2(9);

				double c[4];
				c[0] = -n26 * n11 * n11 * n13 - n23 * n11 * n11 * n16 + n24 * n11 * n13 * n13 + 2 * n23 * n11 * n13 * n14 - 2 * n21 * n11 * n13 * n16 - 2 * n26 * n11 * n14 * n16 - n24 * n11 * n16 * n16 + n17 * n11 - n26 * n12 * n12 * n13 - n23 * n12 * n12 * n16 + n25 * n12 * n13 * n13 + 2 * n23 * n12 * n13 * n15 - 2 * n22 * n12 * n13 * n16 - 2 * n26 * n12 * n15 * n16 - n25 * n12 * n16 * n16 + n18 * n12 + n21 * n13 * n13 * n14 + n22 * n13 * n13 * n15 + n26 * n13 * n14 * n14 + 2 * n24 * n13 * n14 * n16 + n26 * n13 * n15 * n15 + 2 * n25 * n13 * n15 * n16 + n23 * n14 * n14 * n16 - n21 * n14 * n16 * n16 - m11 * n17 * n14 + n23 * n15 * n15 * n16 - n22 * n15 * n16 * n16 - m11 * n18 * n15;
				c[1] = n11 * n27 + n17 * n21 + n12 * n28 + n18 * n22 + n11 * n14 * n23 * n23 + n12 * n15 * n23 * n23 - n13 * n16 * n21 * n21 - n11 * n14 * n26 * n26 - n13 * n16 * n22 * n22 - n12 * n15 * n26 * n26 + n13 * n16 * n24 * n24 + n13 * n16 * n25 * n25 + n13 * n13 * n21 * n24 - n11 * n11 * n23 * n26 + n13 * n13 * n22 * n25 - n12 * n12 * n23 * n26 - n16 * n16 * n21 * n24 + n14 * n14 * n23 * n26 - n16 * n16 * n22 * n25 + n15 * n15 * n23 * n26 - m11 * n14 * n27 - m11 * n17 * n24 - m21 * n14 * n17 - m11 * n15 * n28 - m11 * n18 * n25 - m21 * n15 * n18 - 2 * n11 * n13 * n21 * n26 + 2 * n11 * n13 * n23 * n24 - 2 * n11 * n16 * n21 * n23 + 2 * n13 * n14 * n21 * n23 - 2 * n12 * n13 * n22 * n26 + 2 * n12 * n13 * n23 * n25 - 2 * n12 * n16 * n22 * n23 + 2 * n13 * n15 * n22 * n23 - 2 * n11 * n16 * n24 * n26 + 2 * n13 * n14 * n24 * n26 - 2 * n14 * n16 * n21 * n26 + 2 * n14 * n16 * n23 * n24 - 2 * n12 * n16 * n25 * n26 + 2 * n13 * n15 * n25 * n26 - 2 * n15 * n16 * n22 * n26 + 2 * n15 * n16 * n23 * n25;
				c[2] = n21 * n27 + n22 * n28 + n11 * n23 * n23 * n24 + n14 * n21 * n23 * n23 + n12 * n23 * n23 * n25 - n13 * n21 * n21 * n26 + n15 * n22 * n23 * n23 - n16 * n21 * n21 * n23 - n11 * n24 * n26 * n26 - n13 * n22 * n22 * n26 - n14 * n21 * n26 * n26 - n16 * n22 * n22 * n23 - n12 * n25 * n26 * n26 + n13 * n24 * n24 * n26 - n15 * n22 * n26 * n26 + n16 * n23 * n24 * n24 + n13 * n25 * n25 * n26 + n16 * n23 * n25 * n25 - m11 * n24 * n27 - m21 * n14 * n27 - m21 * n17 * n24 - m11 * n25 * n28 - m21 * n15 * n28 - m21 * n18 * n25 - 2 * n11 * n21 * n23 * n26 + 2 * n13 * n21 * n23 * n24 - 2 * n12 * n22 * n23 * n26 + 2 * n13 * n22 * n23 * n25 + 2 * n14 * n23 * n24 * n26 - 2 * n16 * n21 * n24 * n26 + 2 * n15 * n23 * n25 * n26 - 2 * n16 * n22 * n25 * n26;
				c[3] = -n21 * n21 * n23 * n26 + n21 * n23 * n23 * n24 - n21 * n24 * n26 * n26 - n22 * n22 * n23 * n26 + n22 * n23 * n23 * n25 - n22 * n25 * n26 * n26 + n23 * n24 * n24 * n26 + n23 * n25 * n25 * n26 - m21 * n27 * n24 - m21 * n28 * n25;
				c[3] = 1.0 / c[3];
				double real_roots[3];
				int n_roots = poselib::sturm::solve_cubic_real(c[2] * c[3], c[1] * c[3], c[0] * c[3], real_roots);

				Eigen::Matrix<double, 12, 1> f;
				for (int i = 0; i < n_roots; ++i)
				{
					f = n1 + n2 * real_roots[i];

					FundamentalMatrix model;
					model.descriptor << f[0], f[1], f[2],
						f[3], f[4], f[5],
						f[6], f[7], f[8];
					models_.push_back(model);
				}

				return true;
			}
		}
	}
}