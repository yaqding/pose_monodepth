// Copyright (c) 2025, Yaqing Ding
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
// Author: Yaqing Ding. GB solution to camera pose estimation with different unknown
// focal lengths from four point correspondences with their monocular depths.
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
			class FundamentalMatrixMonoVarGB : public SolverEngine
			{
			public:
				FundamentalMatrixMonoVarGB()
				{
				}

				~FundamentalMatrixMonoVarGB()
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
					return "MonofvarGB";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 4;
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

			OLGA_INLINE bool FundamentalMatrixMonoVarGB::estimateModel(
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

				Eigen::VectorXd coeffs(40);
				coeffs[0] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] - std::pow(d[5], 2);
				coeffs[1] = std::pow(d[8], 2) - 2 * d[8] * d[9] + std::pow(d[9], 2) + std::pow(d[12], 2) - 2 * d[12] * d[13] + std::pow(d[13], 2);
				coeffs[2] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[1] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[5] * d[16] + 2 * d[0] * d[1] * d[17] - 2 * std::pow(d[1], 2) * d[17] + 2 * d[4] * d[5] * d[17] - 2 * std::pow(d[5], 2) * d[17];
				coeffs[3] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[9] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[13] * d[20] - 2 * d[8] * d[9] * d[21] + 2 * std::pow(d[9], 2) * d[21] - 2 * d[12] * d[13] * d[21] + 2 * std::pow(d[13], 2) * d[21];
				coeffs[4] = std::pow(d[20], 2) - 2 * d[20] * d[21] + std::pow(d[21], 2);
				coeffs[5] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[1] * d[16] * d[17] + 2 * d[4] * d[5] * d[16] * d[17] - std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2);
				coeffs[6] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[9] * d[20] * d[21] - 2 * d[12] * d[13] * d[20] * d[21] + std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2);
				coeffs[7] = -std::pow(d[16], 2) + 2 * d[16] * d[17] - std::pow(d[17], 2);
				coeffs[8] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[6] - std::pow(d[6], 2);
				coeffs[9] = std::pow(d[8], 2) - 2 * d[8] * d[10] + std::pow(d[10], 2) + std::pow(d[12], 2) - 2 * d[12] * d[14] + std::pow(d[14], 2);
				coeffs[10] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[2] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[6] * d[16] + 2 * d[0] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[4] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[11] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[10] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[14] * d[20] - 2 * d[8] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[12] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[12] = std::pow(d[20], 2) - 2 * d[20] * d[22] + std::pow(d[22], 2);
				coeffs[13] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[2] * d[16] * d[18] + 2 * d[4] * d[6] * d[16] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
				coeffs[14] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[10] * d[20] * d[22] - 2 * d[12] * d[14] * d[20] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[15] = -std::pow(d[16], 2) + 2 * d[16] * d[18] - std::pow(d[18], 2);
				coeffs[16] = -std::pow(d[0], 2) + 2 * d[0] * d[3] - std::pow(d[3], 2) - std::pow(d[4], 2) + 2 * d[4] * d[7] - std::pow(d[7], 2);
				coeffs[17] = std::pow(d[8], 2) - 2 * d[8] * d[11] + std::pow(d[11], 2) + std::pow(d[12], 2) - 2 * d[12] * d[15] + std::pow(d[15], 2);
				coeffs[18] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[3] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[7] * d[16] + 2 * d[0] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[4] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[19] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[11] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[15] * d[20] - 2 * d[8] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[12] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[20] = std::pow(d[20], 2) - 2 * d[20] * d[23] + std::pow(d[23], 2);
				coeffs[21] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[3] * d[16] * d[19] + 2 * d[4] * d[7] * d[16] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
				coeffs[22] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[11] * d[20] * d[23] - 2 * d[12] * d[15] * d[20] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[23] = -std::pow(d[16], 2) + 2 * d[16] * d[19] - std::pow(d[19], 2);
				coeffs[24] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[5], 2) + 2 * d[5] * d[6] - std::pow(d[6], 2);
				coeffs[25] = std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2) + std::pow(d[13], 2) - 2 * d[13] * d[14] + std::pow(d[14], 2);
				coeffs[26] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[2] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[6] * d[17] + 2 * d[1] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[5] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[27] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[10] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[14] * d[21] - 2 * d[9] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[13] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[28] = std::pow(d[21], 2) - 2 * d[21] * d[22] + std::pow(d[22], 2);
				coeffs[29] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[2] * d[17] * d[18] + 2 * d[5] * d[6] * d[17] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
				coeffs[30] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[10] * d[21] * d[22] - 2 * d[13] * d[14] * d[21] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[31] = -std::pow(d[17], 2) + 2 * d[17] * d[18] - std::pow(d[18], 2);
				coeffs[32] = -std::pow(d[1], 2) + 2 * d[1] * d[3] - std::pow(d[3], 2) - std::pow(d[5], 2) + 2 * d[5] * d[7] - std::pow(d[7], 2);
				coeffs[33] = std::pow(d[9], 2) - 2 * d[9] * d[11] + std::pow(d[11], 2) + std::pow(d[13], 2) - 2 * d[13] * d[15] + std::pow(d[15], 2);
				coeffs[34] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[3] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[7] * d[17] + 2 * d[1] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[5] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[35] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[11] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[15] * d[21] - 2 * d[9] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[13] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[36] = std::pow(d[21], 2) - 2 * d[21] * d[23] + std::pow(d[23], 2);
				coeffs[37] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[3] * d[17] * d[19] + 2 * d[5] * d[7] * d[17] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
				coeffs[38] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[11] * d[21] * d[23] - 2 * d[13] * d[15] * d[21] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[39] = -std::pow(d[17], 2) + 2 * d[17] * d[19] - std::pow(d[19], 2);

				static const int coeffs_ind[] = {0, 8, 16, 24, 32, 1, 9, 17, 25, 33, 0, 8, 16, 24, 32, 2, 10, 8, 0, 18, 16, 24, 26, 32, 34, 3, 11, 1, 9, 19, 17, 25, 27, 33, 35, 9, 1, 17, 25, 33, 4, 12, 20, 28, 36, 2, 10, 0, 18, 16,
												 26, 8, 24, 34, 32, 5, 13, 10, 2, 21, 18, 26, 29, 34, 37, 6, 14, 3, 11, 22, 19, 27, 30, 35, 38, 11, 3, 1, 19, 17, 27, 9, 25, 35, 33, 4, 12, 20, 28, 36, 12, 4, 20, 28, 36, 13, 5, 21, 29, 37,
												 14, 6, 3, 22, 19, 30, 11, 27, 38, 35, 4, 20, 12, 28, 36, 7, 15, 23, 31, 39, 7, 15, 23, 31, 39, 5, 13, 2, 21, 18, 29, 10, 26, 37, 34, 6, 14, 22, 30, 38, 7, 23, 15, 31, 39, 15, 7, 23, 31, 39,
												 5, 21, 13, 29, 37, 6, 22, 14, 30, 38};

				static const int C_ind[] = {0, 1, 6, 13, 19, 20, 21, 26, 33, 39, 42, 44, 49, 51, 57, 60, 61, 63, 65, 66, 68, 72, 73, 76, 79, 80, 81, 82, 84, 86, 89, 91, 93, 97, 99, 103, 105, 108, 112, 116, 120, 121, 126, 133, 139, 142, 144, 147, 149, 150,
											151, 154, 155, 157, 158, 160, 161, 163, 165, 166, 168, 172, 173, 176, 179, 180, 181, 182, 184, 186, 189, 191, 193, 197, 199, 203, 205, 207, 208, 210, 212, 214, 215, 216, 218, 222, 224, 229, 231, 237, 243, 245, 248, 252, 256, 263, 265, 268, 272, 276,
											283, 285, 287, 288, 290, 292, 294, 295, 296, 298, 307, 310, 314, 315, 318, 322, 324, 329, 331, 337, 340, 341, 346, 353, 359, 362, 364, 367, 369, 370, 371, 374, 375, 377, 378, 382, 384, 389, 391, 397, 407, 410, 414, 415, 418, 423, 425, 428, 432, 436,
											447, 450, 454, 455, 458, 467, 470, 474, 475, 478};

				Eigen::MatrixXd C = Eigen::MatrixXd::Zero(20, 24);
				for (int i = 0; i < 160; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				Eigen::MatrixXd C0 = C.leftCols(20);
				Eigen::MatrixXd C1 = C.rightCols(4);
				Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
				Eigen::MatrixXd RR(8, 4);
				RR << -C12.bottomRows(4), Eigen::MatrixXd::Identity(4, 4);

				static const int AM_ind[] = {0, 1, 2, 3};
				Eigen::MatrixXd AM(4, 4);
				for (int i = 0; i < 4; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
				Eigen::ArrayXcd D = es.eigenvalues();
				Eigen::ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).replicate(4, 1)).eval();

				Eigen::MatrixXd sols(5, 4);
				int m = 0;
				for (int k = 0; k < 4; ++k)
				{

					if (abs(D(k).imag()) > 0.001 || V(2, k).real() < 0.0 ||
						V(3, k).real() < 0.0 || abs(V(2, k).imag()) > 0.001 || abs(V(3, k).imag()) > 0.001)
						continue;

					sols(1, m) = D(k).real();	 // u
					sols(2, m) = V(1, k).real(); // v
					sols(3, m) = V(2, k).real(); // f
					sols(4, m) = V(3, k).real(); // m

					double ss = -(coeffs[0] * sols(1, m) * sols(1, m) * sols(3, m) + coeffs[1] * sols(2, m) * sols(2, m) * sols(4, m) + coeffs[2] * sols(1, m) * sols(3, m) + coeffs[3] * sols(2, m) * sols(4, m) + coeffs[5] * sols(3, m) + coeffs[6] * sols(4, m) + coeffs[7]) / coeffs[4];

					if (ss < 0)
						continue;
					sols(0, m) = std::sqrt(ss); // s
					sols(3, m) = 1.0 / std::sqrt(V(2, k).real());
					sols(4, m) = 1.0 / std::sqrt(V(3, k).real() / ss);
					++m;
				}

				sols.conservativeResize(5, m);

				for (size_t k = 0; k < m; ++k)
				{
					double s = sols(0, k);
					double u = sols(1, k);
					double v = sols(2, k);
					double f = sols(3, k);
					double w = sols(4, k);

					Eigen::Matrix3d K1inv;
					K1inv << 1.0 / f, 0, 0,
						0, 1.0 / f, 0,
						0, 0, 1;

					Eigen::Matrix3d K2inv;
					K2inv << 1.0 / w, 0, 0,
						0, 1.0 / w, 0,
						0, 0, 1;

					Eigen::Vector3d v1 = s * (depth2[0] + v) * K2inv * x2.col(0) - s * (depth2[1] + v) * K2inv * x2.col(1);
					Eigen::Vector3d v2 = s * (depth2[0] + v) * K2inv * x2.col(0) - s * (depth2[2] + v) * K2inv * x2.col(2);
					Eigen::Matrix3d Y;
					Y << v1, v2, v1.cross(v2);

					Eigen::Vector3d u1 = (depth1[0] + u) * K1inv * x1.col(0) - (depth1[1] + u) * K1inv * x1.col(1);
					Eigen::Vector3d u2 = (depth1[0] + u) * K1inv * x1.col(0) - (depth1[2] + u) * K1inv * x1.col(2);
					Eigen::Matrix3d X;
					X << u1, u2, u1.cross(u2);
					X = X.inverse().eval();

					Eigen::Matrix3d rot = Y * X;

					Eigen::Vector3d trans1 = (depth1[0] + u) * rot * K1inv * x1.col(0);
					Eigen::Vector3d trans2 = s * (depth2[0] + v) * K2inv * x2.col(0);
					Eigen::Vector3d trans = trans2 - trans1;

					Eigen::Matrix3d TX;
					TX << 0, -trans(2), trans(1),
						trans(2), 0, -trans(0),
						-trans(1), trans(0), 0;

					FundamentalMatrix model;
					model.descriptor = K2inv * TX * rot * K1inv;
					models_.push_back(model);
				}

				return true;
			}
		}
	}
}