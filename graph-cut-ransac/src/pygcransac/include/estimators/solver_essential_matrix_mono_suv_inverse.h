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
// Author: Yaqing Ding. Solution to calibrated camera pose estimation from
// three point correspondences with their monocular depths. (inverse depth model)
// Y. Ding, V. VAVRA, V. Kocur, J. Yang, T. Sattler, Z. Kukelova,
// Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation, arxiv 2025
#pragma once

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialMatrixInverseSolver : public SolverEngine
			{
			public:
				EssentialMatrixInverseSolver()
				{
				}

				~EssentialMatrixInverseSolver()
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
					return "3ptinv";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 10;
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

			using namespace Eigen;

			MatrixXcd solver_p3p_mono_inverse(const VectorXd &data)
			{
				const double *d = data.data();
				VectorXd coeffs(18);
				coeffs[0] = -std::pow(d[6], 2) - std::pow(d[9], 2) - 1;
				coeffs[1] = 2 * d[6] * d[7] + 2 * d[9] * d[10] + 2;
				coeffs[2] = -std::pow(d[7], 2) - std::pow(d[10], 2) - 1;
				coeffs[3] = std::pow(d[1], 2) + std::pow(d[4], 2) + 1;
				coeffs[4] = -2 * d[0] * d[1] - 2 * d[3] * d[4] - 2;
				coeffs[5] = std::pow(d[0], 2) + std::pow(d[3], 2) + 1;
				coeffs[6] = 2 * d[6] * d[8] + 2 * d[9] * d[11] + 2;
				coeffs[7] = -std::pow(d[8], 2) - std::pow(d[11], 2) - 1;
				coeffs[8] = std::pow(d[2], 2) + std::pow(d[5], 2) + 1;
				coeffs[9] = -2 * d[0] * d[2] - 2 * d[3] * d[5] - 2;
				coeffs[10] = 2 * d[7] * d[8] + 2 * d[10] * d[11] + 2;
				coeffs[11] = -2 * d[1] * d[2] - 2 * d[4] * d[5] - 2;
				coeffs[12] = d[15] - d[16];
				coeffs[13] = -d[15] + d[17];
				coeffs[14] = d[16] - d[17];
				coeffs[15] = d[13] - d[14];
				coeffs[16] = d[12] - d[13];
				coeffs[17] = -d[12] + d[14];

				static const int coeffs_ind[] = {1, 2, 2, 12, 2, 2, 1, 6, 10, 13, 14, 2, 10, 14, 7, 7, 7, 7, 3, 3, 15, 3, 15, 11, 15, 11, 15, 2, 2, 1, 6, 10, 12, 14, 2, 2, 6, 7, 7, 13, 7, 10, 14, 7, 7, 1, 2, 12, 2, 2,
												 2, 6, 10, 12, 13, 14, 1, 10, 2, 14, 2, 7, 7, 6, 13, 7, 7, 10, 14, 7, 7, 3, 11, 3, 15, 16, 16, 11, 3, 15, 3, 4, 8, 11, 8, 16, 15, 17, 4, 8, 11, 15, 16, 4, 17, 8, 8, 11, 15, 9,
												 8, 8, 17, 9, 8, 8, 17, 1, 12, 2, 2, 2, 2, 12, 1, 6, 13, 14, 10, 10, 2, 14, 2, 13, 7, 6, 7, 7, 14, 7, 10, 7, 7, 4, 3, 3, 16, 4, 3, 3, 16, 9, 4, 11, 15, 16, 17, 9, 4, 11,
												 15, 16, 9, 16, 11, 17, 4, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 12, 1, 0, 12, 0, 0, 1, 12, 0, 1, 0, 12, 0, 13, 0, 6, 0, 0, 6, 13, 6, 0, 13, 0, 0, 6, 13, 5, 5, 5, 5,
												 4, 16, 5, 5, 9, 17, 5, 5, 9, 8, 8, 17, 5, 5, 5, 4, 16, 5, 5, 9, 17, 5, 5, 9, 8, 8, 17, 5, 5, 5, 5, 4, 16, 5, 9, 5, 17, 5, 9, 17, 8, 8};

				static const int C_ind[] = {0, 1, 6, 7, 54, 59, 110, 111, 114, 115, 116, 164, 167, 169, 220, 222, 273, 275, 325, 330, 377, 380, 387, 438, 477, 491, 534, 550, 563, 605, 606, 617, 619, 634, 659, 662, 715, 717, 725, 727, 768, 770, 781, 823, 824, 880, 885, 890, 891, 934,
											935, 990, 993, 996, 998, 1000, 1016, 1043, 1045, 1052, 1070, 1100, 1101, 1102, 1104, 1151, 1152, 1153, 1158, 1207, 1210, 1263, 1265, 1269, 1294, 1295, 1305, 1310, 1315, 1339, 1340, 1351, 1365, 1371, 1373, 1395, 1400, 1403, 1404, 1416, 1421, 1451, 1452, 1460, 1467, 1471, 1472, 1477, 1487, 1516,
											1532, 1533, 1557, 1569, 1583, 1584, 1614, 1650, 1655, 1657, 1659, 1704, 1705, 1760, 1761, 1762, 1763, 1766, 1767, 1813, 1815, 1817, 1818, 1868, 1875, 1877, 1878, 1921, 1922, 1924, 1926, 1980, 1985, 2008, 2035, 2037, 2050, 2063, 2085, 2088, 2095, 2121, 2133, 2145, 2155, 2156, 2158, 2172, 2176, 2191,
											2206, 2207, 2227, 2243, 2250, 2257, 2258, 2265, 2305, 2310, 2332, 2337, 2396, 2403, 2431, 2434, 2514, 2518, 2521, 2522, 2548, 2550, 2578, 2608, 2610, 2619, 2620, 2646, 2647, 2649, 2654, 2733, 2738, 2741, 2742, 2765, 2767, 2769, 2794, 2828, 2830, 2836, 2852, 2864, 2866, 2870, 2953, 2958, 2980, 2985,
											3007, 3019, 3044, 3051, 3066, 3073, 3079, 3082, 3098, 3117, 3120, 3128, 3162, 3166, 3198, 3216, 3232, 3256, 3258, 3274, 3286, 3294, 3297, 3312, 3325, 3328, 3341, 3381, 3389, 3413, 3415, 3435, 3453, 3478, 3497, 3500, 3507, 3512, 3532, 3539, 3546, 3551};

				MatrixXd C = MatrixXd::Zero(54, 66);
				for (int i = 0; i < 242; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(54);
				MatrixXd C1 = C.rightCols(12);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(24, 12);
				RR << -C12.bottomRows(12), MatrixXd::Identity(12, 12);

				static const int AM_ind[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
				MatrixXd AM(12, 12);
				for (int i = 0; i < 12; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				ArrayXXcd scale = (D.transpose() / (V.row(0) * V.row(0))).sqrt();
				V = (V * scale.replicate(12, 1)).eval();

				MatrixXcd sols(5, 12);
				sols.row(0) = V.row(0);
				sols.row(1) = V.row(4);
				sols.row(2) = V.row(8);
				sols.row(3) = V.row(1).array() / (sols.row(0).array());
				sols.row(4) = V.row(2).array() / (sols.row(0).array());
				return sols;
			}

			OLGA_INLINE bool EssentialMatrixInverseSolver::estimateModel(
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
				double idepth1[3];
				double idepth2[3];

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					x1.col(i) << data_ptr[offset], data_ptr[offset + 1], 1.0;
					x2.col(i) << data_ptr[offset + 2], data_ptr[offset + 3], 1.0;
					idepth1[i] = data_ptr[offset + 4];
					idepth2[i] = data_ptr[offset + 5];
				}

				Eigen::VectorXd datain(18);
				datain << x1(0, 0), x1(0, 1), x1(0, 2), x1(1, 0), x1(1, 1), x1(1, 2), x2(0, 0), x2(0, 1), x2(0, 2), x2(1, 0), x2(1, 1), x2(1, 2), idepth1[0], idepth1[1], idepth1[2], idepth2[0], idepth2[1], idepth2[2];

				Eigen::MatrixXcd sols(5, 12);
				sols = solver_p3p_mono_inverse(datain);

				for (size_t k = 0; k < 12; ++k)
				{

					if (abs(sols(3, k).imag()) > 0.001 || (sols(3, k).real()) <= 0.001 ||
						abs(sols(4, k).imag()) > 0.001 || (sols(4, k).real()) <= 0.001 ||
						!((sols(0, k).real() > 0.0 && sols(1, k).real() > 0.0 && sols(2, k).real() > 0.0) || (sols(0, k).real() < 0.0 && sols(1, k).real() < 0.0 && sols(2, k).real() < 0.0)))
						continue;

					double b1 = abs(sols(0, k).real());
					double b2 = abs(sols(1, k).real());
					double b3 = abs(sols(2, k).real());
					double a2 = (sols(3, k).real());
					double a3 = (sols(4, k).real());

					Eigen::Vector3d v1 = b1 * x2.col(0) - b2 * x2.col(1);
					Eigen::Vector3d v2 = b1 * x2.col(0) - b3 * x2.col(2);
					Eigen::Matrix3d Y;
					Y << v1, v2, v1.cross(v2);

					Eigen::Vector3d u1 = x1.col(0) - a2 * x1.col(1);
					Eigen::Vector3d u2 = x1.col(0) - a3 * x1.col(2);
					Eigen::Matrix3d X;
					X << u1, u2, u1.cross(u2);
					X = X.inverse().eval();

					Eigen::Matrix3d rot = Y * X;

					Eigen::Vector3d trans1 = rot * x1.col(0);
					Eigen::Vector3d trans2 = b1 * x2.col(0);
					Eigen::Vector3d trans = trans2 - trans1;

					Eigen::Matrix3d TX;
					TX << 0, -trans(2), trans(1),
						trans(2), 0, -trans(0),
						-trans(1), trans(0), 0;

					Eigen::Matrix<double, 3, 3> Ess;
					Ess = TX * rot;

					EssentialMatrix model;
					model.descriptor = Ess;
					models_.push_back(model);
				}

				return models_.size();
			}
		}
	}
}