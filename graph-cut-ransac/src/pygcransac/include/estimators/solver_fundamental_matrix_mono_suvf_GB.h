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
// Author: Yaqing Ding. GB solution to camera pose estimation with equal unknown
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
			class FundamentalMatrixMonoEqualGB : public SolverEngine
			{
			public:
				FundamentalMatrixMonoEqualGB()
				{
				}

				~FundamentalMatrixMonoEqualGB()
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
					return "MonofGB";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 8;
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

			OLGA_INLINE bool FundamentalMatrixMonoEqualGB::estimateModel(
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

				Eigen::VectorXd coeffs(32);
				coeffs[0] = std::pow(d[8], 2) - 2 * d[8] * d[9] + std::pow(d[9], 2) + std::pow(d[12], 2) - 2 * d[12] * d[13] + std::pow(d[13], 2);
				coeffs[1] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[9] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[13] * d[20] - 2 * d[8] * d[9] * d[21] + 2 * std::pow(d[9], 2) * d[21] - 2 * d[12] * d[13] * d[21] + 2 * std::pow(d[13], 2) * d[21];
				coeffs[2] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] - std::pow(d[5], 2);
				coeffs[3] = std::pow(d[20], 2) - 2 * d[20] * d[21] + std::pow(d[21], 2);
				coeffs[4] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[9] * d[20] * d[21] - 2 * d[12] * d[13] * d[20] * d[21] + std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2);
				coeffs[5] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[1] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[5] * d[16] + 2 * d[0] * d[1] * d[17] - 2 * std::pow(d[1], 2) * d[17] + 2 * d[4] * d[5] * d[17] - 2 * std::pow(d[5], 2) * d[17];
				coeffs[6] = -std::pow(d[16], 2) + 2 * d[16] * d[17] - std::pow(d[17], 2);
				coeffs[7] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[1] * d[16] * d[17] + 2 * d[4] * d[5] * d[16] * d[17] - std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2);
				coeffs[8] = std::pow(d[8], 2) - 2 * d[8] * d[10] + std::pow(d[10], 2) + std::pow(d[12], 2) - 2 * d[12] * d[14] + std::pow(d[14], 2);
				coeffs[9] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[10] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[14] * d[20] - 2 * d[8] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[12] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[10] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[6] - std::pow(d[6], 2);
				coeffs[11] = std::pow(d[20], 2) - 2 * d[20] * d[22] + std::pow(d[22], 2);
				coeffs[12] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[10] * d[20] * d[22] - 2 * d[12] * d[14] * d[20] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[13] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[2] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[6] * d[16] + 2 * d[0] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[4] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[14] = -std::pow(d[16], 2) + 2 * d[16] * d[18] - std::pow(d[18], 2);
				coeffs[15] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[2] * d[16] * d[18] + 2 * d[4] * d[6] * d[16] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
				coeffs[16] = std::pow(d[8], 2) - 2 * d[8] * d[11] + std::pow(d[11], 2) + std::pow(d[12], 2) - 2 * d[12] * d[15] + std::pow(d[15], 2);
				coeffs[17] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[11] * d[20] + 2 * std::pow(d[12], 2) * d[20] - 2 * d[12] * d[15] * d[20] - 2 * d[8] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] - 2 * d[12] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
				coeffs[18] = -std::pow(d[0], 2) + 2 * d[0] * d[3] - std::pow(d[3], 2) - std::pow(d[4], 2) + 2 * d[4] * d[7] - std::pow(d[7], 2);
				coeffs[19] = std::pow(d[20], 2) - 2 * d[20] * d[23] + std::pow(d[23], 2);
				coeffs[20] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) - 2 * d[8] * d[11] * d[20] * d[23] - 2 * d[12] * d[15] * d[20] * d[23] + std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
				coeffs[21] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[3] * d[16] - 2 * std::pow(d[4], 2) * d[16] + 2 * d[4] * d[7] * d[16] + 2 * d[0] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] + 2 * d[4] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
				coeffs[22] = -std::pow(d[16], 2) + 2 * d[16] * d[19] - std::pow(d[19], 2);
				coeffs[23] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) + 2 * d[0] * d[3] * d[16] * d[19] + 2 * d[4] * d[7] * d[16] * d[19] - std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
				coeffs[24] = std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2) + std::pow(d[13], 2) - 2 * d[13] * d[14] + std::pow(d[14], 2);
				coeffs[25] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[10] * d[21] + 2 * std::pow(d[13], 2) * d[21] - 2 * d[13] * d[14] * d[21] - 2 * d[9] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] - 2 * d[13] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
				coeffs[26] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[5], 2) + 2 * d[5] * d[6] - std::pow(d[6], 2);
				coeffs[27] = std::pow(d[21], 2) - 2 * d[21] * d[22] + std::pow(d[22], 2);
				coeffs[28] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) - 2 * d[9] * d[10] * d[21] * d[22] - 2 * d[13] * d[14] * d[21] * d[22] + std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
				coeffs[29] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[2] * d[17] - 2 * std::pow(d[5], 2) * d[17] + 2 * d[5] * d[6] * d[17] + 2 * d[1] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] + 2 * d[5] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
				coeffs[30] = -std::pow(d[17], 2) + 2 * d[17] * d[18] - std::pow(d[18], 2);
				coeffs[31] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) + 2 * d[1] * d[2] * d[17] * d[18] + 2 * d[5] * d[6] * d[17] * d[18] - std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);

				static const int coeffs_ind[] = {0, 8, 16, 24, 0, 8, 16, 24, 0, 8, 16, 24, 1, 0, 9, 8, 17, 16, 25, 24, 2, 10, 18, 26, 2, 10, 18, 26, 5, 2, 13, 10, 21, 18, 29, 26, 2, 10, 18, 26, 6, 5, 14, 13, 21, 22, 29, 30, 2, 18,
												 10, 26, 3, 19, 11, 27, 6, 14, 22, 30, 7, 5, 15, 13, 23, 21, 31, 29, 5, 13, 2, 21, 18, 10, 26, 29, 4, 20, 3, 19, 12, 11, 27, 28, 6, 14, 5, 21, 22, 13, 29, 30, 4, 20, 12, 28, 7, 15, 5, 23,
												 21, 13, 29, 31, 1, 9, 17, 0, 16, 8, 25, 24, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 6, 22, 14, 30, 7, 6, 15, 14, 23, 22, 31, 30, 1, 9, 17, 0, 16, 8, 24, 25, 4, 1, 12, 9, 20, 17,
												 28, 25, 4, 3, 12, 11, 20, 1, 17, 19, 9, 28, 25, 27, 4, 12, 20, 1, 17, 9, 25, 28, 4, 12, 20, 28, 7, 23, 6, 22, 15, 14, 30, 31, 7, 23, 15, 31, 7, 15, 23, 31};

				static const int C_ind[] = {0, 4, 11, 20, 25, 29, 32, 43, 50, 54, 60, 71, 72, 75, 76, 79, 83, 86, 92, 94, 96, 100, 107, 116, 121, 125, 128, 139, 144, 146, 148, 150, 155, 156, 164, 167, 171, 175, 182, 190, 192, 193, 196, 197, 200, 203, 211, 212, 225, 226,
											232, 237, 249, 250, 256, 261, 265, 269, 272, 283, 288, 290, 292, 294, 299, 300, 308, 311, 315, 319, 325, 326, 327, 329, 330, 334, 345, 346, 349, 351, 352, 353, 354, 357, 363, 367, 369, 370, 374, 376, 381, 382, 397, 399, 401, 402, 411, 415, 421, 422,
											423, 425, 426, 430, 433, 437, 440, 441, 442, 448, 451, 453, 456, 460, 467, 476, 481, 485, 488, 499, 507, 511, 518, 526, 537, 538, 544, 549, 553, 554, 557, 558, 560, 564, 571, 575, 578, 582, 588, 589, 591, 593, 594, 599, 600, 603, 604, 607, 611, 614,
											620, 622, 625, 626, 629, 630, 632, 633, 634, 636, 640, 643, 645, 647, 650, 654, 660, 661, 663, 665, 666, 671, 675, 679, 686, 694, 705, 706, 709, 711, 712, 713, 714, 717, 733, 735, 737, 738, 746, 750, 756, 767};

				Eigen::MatrixXd C = Eigen::MatrixXd::Zero(24, 32);
				for (int i = 0; i < 192; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				Eigen::MatrixXd C0 = C.leftCols(24);
				Eigen::MatrixXd C1 = C.rightCols(8);
				Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
				Eigen::MatrixXd RR(14, 8);
				RR << -C12.bottomRows(6), Eigen::MatrixXd::Identity(8, 8);

				static const int AM_ind[] = {0, 1, 2, 8, 3, 4, 11, 5};
				Eigen::MatrixXd AM(8, 8);
				for (int i = 0; i < 8; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
				Eigen::ArrayXcd D = es.eigenvalues();
				Eigen::ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(6).replicate(8, 1)).eval();

				Eigen::MatrixXd sols(4, 8);
				int m = 0;
				for (int k = 0; k < 8; ++k)
				{

					if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0 ||
						abs(V(7, k).imag()) > 0.001)
						continue;

					sols(3, m) = std::sqrt(D(k).real()); // f
					sols(2, m) = V(7, k).real();		 // v
					double v2 = sols(2, m) * sols(2, m);
					Eigen::MatrixXd A1(3, 3);
					A1 << coeffs[0] * v2 + coeffs[1] * sols(2, m) + coeffs[3] * D(k).real() + coeffs[4], coeffs[2], coeffs[5],
						coeffs[8] * v2 + coeffs[9] * sols(2, m) + coeffs[11] * D(k).real() + coeffs[12], coeffs[10], coeffs[13],
						coeffs[16] * v2 + coeffs[17] * sols(2, m) + coeffs[19] * D(k).real() + coeffs[20], coeffs[18], coeffs[21];
					Eigen::VectorXd A0(3, 1);
					A0 << -(coeffs[6] * D(k).real() + coeffs[7]),
						-(coeffs[14] * D(k).real() + coeffs[15]),
						-(coeffs[22] * D(k).real() + coeffs[23]);
					Eigen::VectorXd xz = A1.partialPivLu().solve(A0);
					if (xz[0] < 0)
						continue;
					sols(0, m) = std::sqrt(xz[0]); // s
					sols(1, m) = xz[2];			   // u
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