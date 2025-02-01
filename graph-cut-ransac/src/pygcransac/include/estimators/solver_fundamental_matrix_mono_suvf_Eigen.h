// Copyright (c) 2025, Yaqing Ding.
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
// Author: Yaqing Ding. PEP solution to camera pose estimation with equal unknown
// focal length from four point correspondences with their monocular depths.
// Y. Ding, V. Vávra, V. Kocur, J. Yang, T. Sattler, Z. Kukelova,
// Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation, arxiv 2025
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
			class FundamentalMatrixMonoEqualEigen : public SolverEngine
			{
			public:
				FundamentalMatrixMonoEqualEigen()
				{
				}

				~FundamentalMatrixMonoEqualEigen()
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
					return "MonofE";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 2;
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

			OLGA_INLINE bool FundamentalMatrixMonoEqualEigen::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				Eigen::MatrixXd x1(3, 4);
				Eigen::MatrixXd x2(3, 4);
				double depth1[4];
				double depth2[4];

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					x1.col(i) << data_ptr[offset], data_ptr[offset + 1], 1.0;
					x2.col(i) << data_ptr[offset + 2], data_ptr[offset + 3], 1.0;
					depth1[i] = data_ptr[offset + 4];
					depth2[i] = data_ptr[offset + 5];
				}
				Eigen::VectorXd d(24);
				d << x1(0, 0), x1(0, 1), x1(0, 2), x1(0, 3), x1(1, 0), x1(1, 1), x1(1, 2), x1(1, 3), x2(0, 0), x2(0, 1), x2(0, 2), x2(0, 3), x2(1, 0), x2(1, 1), x2(1, 2), x2(1, 3), depth1[0], depth1[1], depth1[2], depth1[3], depth2[0], depth2[1], depth2[2], depth2[3];

				Eigen::VectorXd coeffs(48);
				coeffs[0] = std::pow(d[8], 2) - 2 * d[8] * d[9] + std::pow(d[9], 2) + std::pow(d[12], 2) - 2 * d[12] * d[13] + std::pow(d[13], 2);
				coeffs[1] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] - std::pow(d[5], 2);
				coeffs[2] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[9] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[13] * d[20] - 2 * d[8] * d[9] * d[21] + 2 * std::pow(d[9], 2) * d[21] - 2 * d[12] * d[13] * d[21] + 2 * std::pow(d[13], 2) * d[21];
				coeffs[3] = std::pow(d[20], 2) - 2 * d[20] * d[21] + std::pow(d[21], 2);
				coeffs[4] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[9] * d[20] * d[21] - 2 * d[12] * d[13] * d[20] * d[21] + std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2);
				coeffs[5] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[1] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[5] * d[16] + 2 * d[0] * d[1] * d[17] - 2 * std::pow(d[1], 2) * d[17] + 2 * d[4] * d[5] * d[17] - 2 * std::pow(d[5], 2) * d[17];
				coeffs[6] = -std::pow(d[16], 2) + 2 * d[16] * d[17] - std::pow(d[17], 2);
				coeffs[7] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[1] * d[16] * d[17] + 2 * d[4] * d[5] * d[16] * d[17] - std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2);
				coeffs[8] = std::pow(d[8], 2) - 2 * d[8] * d[10] + std::pow(d[10], 2) + std::pow(d[12], 2) - 2 * d[12] * d[14] + std::pow(d[14], 2);
				coeffs[9] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[6] - std::pow(d[6], 2);
				coeffs[10] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[10] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[14] * d[20] - 2 * d[8] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[12] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[11] = std::pow(d[20], 2) - 2 * d[20] * d[22] + std::pow(d[22], 2);
				coeffs[12] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[10] * d[20] * d[22] - 2 * d[12] * d[14] * d[20] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[13] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[2] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[6] * d[16] + 2 * d[0] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[4] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[14] = -std::pow(d[16], 2) + 2 * d[16] * d[18] - std::pow(d[18], 2);
				coeffs[15] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[2] * d[16] * d[18] + 2 * d[4] * d[6] * d[16] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
				coeffs[16] = std::pow(d[8], 2) - 2 * d[8] * d[11] + std::pow(d[11], 2) + std::pow(d[12], 2) - 2 * d[12] * d[15] + std::pow(d[15], 2);
				coeffs[17] = -std::pow(d[0], 2) + 2 * d[0] * d[3] - std::pow(d[3], 2) - std::pow(d[4], 2) + 2 * d[4] * d[7] - std::pow(d[7], 2);
				coeffs[18] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[11] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[15] * d[20] - 2 * d[8] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[12] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[19] = std::pow(d[20], 2) - 2 * d[20] * d[23] + std::pow(d[23], 2);
				coeffs[20] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[11] * d[20] * d[23] - 2 * d[12] * d[15] * d[20] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[21] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[3] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[7] * d[16] + 2 * d[0] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[4] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[22] = -std::pow(d[16], 2) + 2 * d[16] * d[19] - std::pow(d[19], 2);
				coeffs[23] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[3] * d[16] * d[19] + 2 * d[4] * d[7] * d[16] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
				coeffs[24] = std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2) + std::pow(d[13], 2) - 2 * d[13] * d[14] + std::pow(d[14], 2);
				coeffs[25] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[5], 2) + 2 * d[5] * d[6] - std::pow(d[6], 2);
				coeffs[26] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[10] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[14] * d[21] - 2 * d[9] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[13] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[27] = std::pow(d[21], 2) - 2 * d[21] * d[22] + std::pow(d[22], 2);
				coeffs[28] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[10] * d[21] * d[22] - 2 * d[13] * d[14] * d[21] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[29] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[2] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[6] * d[17] + 2 * d[1] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[5] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[30] = -std::pow(d[17], 2) + 2 * d[17] * d[18] - std::pow(d[18], 2);
				coeffs[31] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[2] * d[17] * d[18] + 2 * d[5] * d[6] * d[17] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
				coeffs[32] = std::pow(d[9], 2) - 2 * d[9] * d[11] + std::pow(d[11], 2) + std::pow(d[13], 2) - 2 * d[13] * d[15] + std::pow(d[15], 2);
				coeffs[33] = -std::pow(d[1], 2) + 2 * d[1] * d[3] - std::pow(d[3], 2) - std::pow(d[5], 2) + 2 * d[5] * d[7] - std::pow(d[7], 2);
				coeffs[34] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[11] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[15] * d[21] - 2 * d[9] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[13] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[35] = std::pow(d[21], 2) - 2 * d[21] * d[23] + std::pow(d[23], 2);
				coeffs[36] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[11] * d[21] * d[23] - 2 * d[13] * d[15] * d[21] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[37] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[3] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[7] * d[17] + 2 * d[1] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[5] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[38] = -std::pow(d[17], 2) + 2 * d[17] * d[19] - std::pow(d[19], 2);
				coeffs[39] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[3] * d[17] * d[19] + 2 * d[5] * d[7] * d[17] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
				coeffs[40] = std::pow(d[10], 2) - 2 * d[10] * d[11] + std::pow(d[11], 2) + std::pow(d[14], 2) - 2 * d[14] * d[15] + std::pow(d[15], 2);
				coeffs[41] = -std::pow(d[2], 2) + 2 * d[2] * d[3] - std::pow(d[3], 2) - std::pow(d[6], 2) + 2 * d[6] * d[7] - std::pow(d[7], 2);
				coeffs[42] = 2 * std::pow(d[10], 2) * d[22] - 2 * d[10] * d[11] * d[22] + 2 * std::pow(d[14], 2) * d[22] - 2 * d[14] * d[15] * d[22] - 2 * d[10] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[14] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[43] = std::pow(d[22], 2) - 2 * d[22] * d[23] + std::pow(d[23], 2);
				coeffs[44] = std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2) - 2 * d[10] * d[11] * d[22] * d[23] - 2 * d[14] * d[15] * d[22] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[45] = -2 * std::pow(d[2], 2) * d[18] + 2 * d[2] * d[3] * d[18] - 2 * std::pow(d[6], 2) * d[18] + 2 * d[6] * d[7] * d[18] + 2 * d[2] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[6] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[46] = -std::pow(d[18], 2) + 2 * d[18] * d[19] - std::pow(d[19], 2);
				coeffs[47] = -std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2) + 2 * d[2] * d[3] * d[18] * d[19] + 2 * d[6] * d[7] * d[18] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);

				Eigen::MatrixXd C0(6, 6);
				C0 << coeffs[0], coeffs[1], coeffs[2], coeffs[4], coeffs[5], coeffs[7],
					coeffs[8], coeffs[9], coeffs[10], coeffs[12], coeffs[13], coeffs[15],
					coeffs[16], coeffs[17], coeffs[18], coeffs[20], coeffs[21], coeffs[23],
					coeffs[24], coeffs[25], coeffs[26], coeffs[28], coeffs[29], coeffs[31],
					coeffs[32], coeffs[33], coeffs[34], coeffs[36], coeffs[37], coeffs[39],
					coeffs[40], coeffs[41], coeffs[42], coeffs[44], coeffs[45], coeffs[47];

				Eigen::MatrixXd C1(6, 2);
				C1 << coeffs[3], coeffs[6],
					coeffs[11], coeffs[14],
					coeffs[19], coeffs[22],
					coeffs[27], coeffs[30],
					coeffs[35], coeffs[38],
					coeffs[43], coeffs[46];

				Eigen::MatrixXd C12 = -C0.partialPivLu().solve(C1);
				Eigen::MatrixXd AM(2, 2);
				AM << C12(3, 0), C12(3, 1),
					C12(5, 0), C12(5, 1);

				Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
				Eigen::ArrayXcd D = es.eigenvalues();
				Eigen::ArrayXXcd V = es.eigenvectors();

				Eigen::MatrixXcd V0(1, 2);
				V0 = V.row(0) / V.row(1);

				Eigen::MatrixXd sols(4, 2);
				int m = 0;
				for (int k = 0; k < 2; ++k)
				{

					if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0 ||
						abs(V0(0, k).imag()) > 0.001 || V0(0, k).real() < 0.0)
						continue;

					double cc = V0(0, k).real(); // c x
					Eigen::MatrixXd A1(4, 4);
					A1 << coeffs[0] * cc, coeffs[1], coeffs[2] * cc, coeffs[5],
						coeffs[8] * cc, coeffs[9], coeffs[10] * cc, coeffs[13],
						coeffs[16] * cc, coeffs[17], coeffs[18] * cc, coeffs[21],
						coeffs[24] * cc, coeffs[25], coeffs[26] * cc, coeffs[29];
					Eigen::VectorXd A0(4, 1);
					A0 << -(coeffs[3] * cc / D(k).real() + coeffs[4] * cc + coeffs[6] / D(k).real() + coeffs[7]),
						-(coeffs[11] * cc / D(k).real() + coeffs[12] * cc + coeffs[14] / D(k).real() + coeffs[15]),
						-(coeffs[19] * cc / D(k).real() + coeffs[20] * cc + coeffs[22] / D(k).real() + coeffs[23]),
						-(coeffs[27] * cc / D(k).real() + coeffs[28] * cc + coeffs[30] / D(k).real() + coeffs[31]);

					Eigen::VectorXd xz = A1.partialPivLu().solve(A0);
					sols(1, m) = xz[3];						   // u
					sols(2, m) = xz[2];						   // v
					sols(0, m) = std::sqrt(V0(0, k).real());   // c x
					sols(3, m) = 1.0 / std::sqrt(D(k).real()); // f
					++m;
				}

				sols.conservativeResize(4, m);

				for (size_t k = 0; k < m; ++k)
				{

					double s = sols(0, k);
					double u = sols(1, k);
					double v = sols(2, k);
					double f = sols(3, k);

					Eigen::Matrix3d Kinv;
					Kinv << 1.0 / f, 0, 0,
						0, 1.0 / f, 0,
						0, 0, 1;

					Eigen::Vector3d v1 = s * (depth2[0] + v) * Kinv * x2.col(0) - s * (depth2[1] + v) * Kinv * x2.col(1);
					Eigen::Vector3d v2 = s * (depth2[0] + v) * Kinv * x2.col(0) - s * (depth2[2] + v) * Kinv * x2.col(2);
					Eigen::Matrix3d Y;
					Y << v1, v2, v1.cross(v2);

					Eigen::Vector3d u1 = (depth1[0] + u) * Kinv * x1.col(0) - (depth1[1] + u) * Kinv * x1.col(1);
					Eigen::Vector3d u2 = (depth1[0] + u) * Kinv * x1.col(0) - (depth1[2] + u) * Kinv * x1.col(2);
					Eigen::Matrix3d X;
					X << u1, u2, u1.cross(u2);
					X = X.inverse().eval();

					Eigen::Matrix3d rot = Y * X;

					Eigen::Vector3d trans1 = (depth1[0] + u) * rot * Kinv * x1.col(0);
					Eigen::Vector3d trans2 = s * (depth2[0] + v) * Kinv * x2.col(0);
					Eigen::Vector3d trans = trans2 - trans1;

					Eigen::Matrix3d TX;
					TX << 0, -trans(2), trans(1),
						trans(2), 0, -trans(0),
						-trans(1), trans(0), 0;

					Eigen::Matrix3d F1;
					F1 = Kinv * TX * rot * Kinv;

					FundamentalMatrix model;
					model.descriptor = F1;
					models_.push_back(model);
				}

				return models_.size();
			}
		}
	}
}