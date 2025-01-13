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
// Author: Yaqing Ding. Grobner basis solution to calibrated camera pose estimation from
// three point correspondences with their monocular depths
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
			class EssentialMatrixThreeMonoSolver : public SolverEngine
			{
			public:
				EssentialMatrixThreeMonoSolver()
				{
				}

				~EssentialMatrixThreeMonoSolver()
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
					return "mono3";
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

			Eigen::MatrixXcd solver_p3p_mono_3d(const Eigen::VectorXd &data)
			{

				const double *d = data.data();
				Eigen::VectorXd coeffs(18);
				coeffs[0] = std::pow(d[6], 2) - 2 * d[6] * d[7] + std::pow(d[7], 2) + std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2);
				coeffs[1] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[3], 2) + 2 * d[3] * d[4] - std::pow(d[4], 2);
				coeffs[2] = 2 * std::pow(d[6], 2) * d[15] - 2 * d[6] * d[7] * d[15] + 2 * std::pow(d[9], 2) * d[15] - 2 * d[9] * d[10] * d[15] - 2 * d[6] * d[7] * d[16] + 2 * std::pow(d[7], 2) * d[16] - 2 * d[9] * d[10] * d[16] + 2 * std::pow(d[10], 2) * d[16];
				coeffs[3] = std::pow(d[6], 2) * std::pow(d[15], 2) + std::pow(d[9], 2) * std::pow(d[15], 2) - 2 * d[6] * d[7] * d[15] * d[16] - 2 * d[9] * d[10] * d[15] * d[16] + std::pow(d[7], 2) * std::pow(d[16], 2) + std::pow(d[10], 2) * std::pow(d[16], 2) + std::pow(d[15], 2) - 2 * d[15] * d[16] + std::pow(d[16], 2);
				coeffs[4] = -2 * std::pow(d[0], 2) * d[12] + 2 * d[0] * d[1] * d[12] - 2 * std::pow(d[3], 2) * d[12] + 2 * d[3] * d[4] * d[12] + 2 * d[0] * d[1] * d[13] - 2 * std::pow(d[1], 2) * d[13] + 2 * d[3] * d[4] * d[13] - 2 * std::pow(d[4], 2) * d[13];
				coeffs[5] = -std::pow(d[0], 2) * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[12], 2) + 2 * d[0] * d[1] * d[12] * d[13] + 2 * d[3] * d[4] * d[12] * d[13] - std::pow(d[1], 2) * std::pow(d[13], 2) - std::pow(d[4], 2) * std::pow(d[13], 2) - std::pow(d[12], 2) + 2 * d[12] * d[13] - std::pow(d[13], 2);
				coeffs[6] = std::pow(d[6], 2) - 2 * d[6] * d[8] + std::pow(d[8], 2) + std::pow(d[9], 2) - 2 * d[9] * d[11] + std::pow(d[11], 2);
				coeffs[7] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[3], 2) + 2 * d[3] * d[5] - std::pow(d[5], 2);
				coeffs[8] = 2 * std::pow(d[6], 2) * d[15] - 2 * d[6] * d[8] * d[15] + 2 * std::pow(d[9], 2) * d[15] - 2 * d[9] * d[11] * d[15] - 2 * d[6] * d[8] * d[17] + 2 * std::pow(d[8], 2) * d[17] - 2 * d[9] * d[11] * d[17] + 2 * std::pow(d[11], 2) * d[17];
				coeffs[9] = std::pow(d[6], 2) * std::pow(d[15], 2) + std::pow(d[9], 2) * std::pow(d[15], 2) - 2 * d[6] * d[8] * d[15] * d[17] - 2 * d[9] * d[11] * d[15] * d[17] + std::pow(d[8], 2) * std::pow(d[17], 2) + std::pow(d[11], 2) * std::pow(d[17], 2) + std::pow(d[15], 2) - 2 * d[15] * d[17] + std::pow(d[17], 2);
				coeffs[10] = -2 * std::pow(d[0], 2) * d[12] + 2 * d[0] * d[2] * d[12] - 2 * std::pow(d[3], 2) * d[12] + 2 * d[3] * d[5] * d[12] + 2 * d[0] * d[2] * d[14] - 2 * std::pow(d[2], 2) * d[14] + 2 * d[3] * d[5] * d[14] - 2 * std::pow(d[5], 2) * d[14];
				coeffs[11] = -std::pow(d[0], 2) * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[12], 2) + 2 * d[0] * d[2] * d[12] * d[14] + 2 * d[3] * d[5] * d[12] * d[14] - std::pow(d[2], 2) * std::pow(d[14], 2) - std::pow(d[5], 2) * std::pow(d[14], 2) - std::pow(d[12], 2) + 2 * d[12] * d[14] - std::pow(d[14], 2);
				coeffs[12] = std::pow(d[7], 2) - 2 * d[7] * d[8] + std::pow(d[8], 2) + std::pow(d[10], 2) - 2 * d[10] * d[11] + std::pow(d[11], 2);
				coeffs[13] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] - std::pow(d[5], 2);
				coeffs[14] = 2 * std::pow(d[7], 2) * d[16] - 2 * d[7] * d[8] * d[16] + 2 * std::pow(d[10], 2) * d[16] - 2 * d[10] * d[11] * d[16] - 2 * d[7] * d[8] * d[17] + 2 * std::pow(d[8], 2) * d[17] - 2 * d[10] * d[11] * d[17] + 2 * std::pow(d[11], 2) * d[17];
				coeffs[15] = std::pow(d[7], 2) * std::pow(d[16], 2) + std::pow(d[10], 2) * std::pow(d[16], 2) - 2 * d[7] * d[8] * d[16] * d[17] - 2 * d[10] * d[11] * d[16] * d[17] + std::pow(d[8], 2) * std::pow(d[17], 2) + std::pow(d[11], 2) * std::pow(d[17], 2) + std::pow(d[16], 2) - 2 * d[16] * d[17] + std::pow(d[17], 2);
				coeffs[16] = -2 * std::pow(d[1], 2) * d[13] + 2 * d[1] * d[2] * d[13] - 2 * std::pow(d[4], 2) * d[13] + 2 * d[4] * d[5] * d[13] + 2 * d[1] * d[2] * d[14] - 2 * std::pow(d[2], 2) * d[14] + 2 * d[4] * d[5] * d[14] - 2 * std::pow(d[5], 2) * d[14];
				coeffs[17] = -std::pow(d[1], 2) * std::pow(d[13], 2) - std::pow(d[4], 2) * std::pow(d[13], 2) + 2 * d[1] * d[2] * d[13] * d[14] + 2 * d[4] * d[5] * d[13] * d[14] - std::pow(d[2], 2) * std::pow(d[14], 2) - std::pow(d[5], 2) * std::pow(d[14], 2) - std::pow(d[13], 2) + 2 * d[13] * d[14] - std::pow(d[14], 2);

				static const int coeffs_ind[] = {0, 6, 12, 1, 7, 13, 2, 8, 0, 6, 12, 14, 6, 0, 12, 1, 7, 13, 3, 9, 2, 8, 14, 15, 4, 10, 7, 1, 16, 13, 8, 2, 6, 12, 0, 14, 9, 3, 8, 14, 2, 15, 3, 9, 15, 4, 10, 16, 7, 13,
												 1, 5, 11, 10, 4, 17, 16, 11, 17, 5, 9, 15, 3, 5, 11, 17, 10, 16, 4, 11, 5, 17};

				static const int C_ind[] = {0, 1, 9, 12, 13, 21, 24, 25, 26, 28, 29, 33, 39, 42, 47, 50, 52, 53, 60, 61, 62, 64, 65, 69, 72, 73, 75, 78, 81, 83, 87, 90, 91, 92, 94, 95, 99, 102, 103, 104, 106, 107, 110, 112, 113, 122, 124, 125, 127, 128,
											130, 132, 133, 135, 138, 141, 143, 151, 152, 154, 163, 164, 166, 170, 172, 173, 175, 176, 178, 183, 186, 191};

				Eigen::MatrixXd C = Eigen::MatrixXd::Zero(12, 16);
				for (int i = 0; i < 72; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				Eigen::MatrixXd C0 = C.leftCols(12);
				Eigen::MatrixXd C1 = C.rightCols(4);
				Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
				Eigen::MatrixXd RR(7, 4);
				RR << -C12.bottomRows(3), Eigen::MatrixXd::Identity(4, 4);

				static const int AM_ind[] = {5, 0, 1, 2};
				Eigen::MatrixXd AM(4, 4);
				for (int i = 0; i < 4; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
				Eigen::ArrayXcd D = es.eigenvalues();
				Eigen::ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).replicate(4, 1)).eval();

				Eigen::MatrixXcd sols(3, 4);
				sols.row(0) = V.row(1);
				sols.row(1) = D.transpose();
				sols.row(2) = V.row(3);
				return sols;
			}

			OLGA_INLINE bool EssentialMatrixThreeMonoSolver::estimateModel(
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
				double depth1[3];
				double depth2[3];

				size_t row = 0;
				for (int i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					x1.col(i) << data_ptr[offset], data_ptr[offset + 1], 1.0;
					x2.col(i) << data_ptr[offset + 2], data_ptr[offset + 3], 1.0;
					depth1[i] = 1.0 / data_ptr[offset + 4];
					depth2[i] = 1.0 / data_ptr[offset + 5];
				}
				Eigen::VectorXd datain(18);
				datain << x1(0, 0), x1(0, 1), x1(0, 2), x1(1, 0), x1(1, 1), x1(1, 2), x2(0, 0), x2(0, 1), x2(0, 2), x2(1, 0), x2(1, 1), x2(1, 2), depth1[0], depth1[1], depth1[2], depth2[0], depth2[1], depth2[2];

				Eigen::MatrixXcd sols(3, 4);
				sols = solver_p3p_mono_3d(datain);

				for (size_t k = 0; k < 4; ++k)
				{

					if (abs(sols(2, k).imag()) > 0.001 ||
						abs(sols(1, k).imag()) > 0.001 ||
						sols(0, k).real() < 0.0)
						continue;

					double s = std::sqrt(sols(0, k).real());
					double u = sols(1, k).real();
					double v = sols(2, k).real();

					Eigen::Vector3d v1 = s * (depth2[0] + v) * x2.col(0) - s * (depth2[1] + v) * x2.col(1);
					Eigen::Vector3d v2 = s * (depth2[0] + v) * x2.col(0) - s * (depth2[2] + v) * x2.col(2);
					Eigen::Matrix3d Y;
					Y << v1, v2, v1.cross(v2);

					Eigen::Vector3d u1 = (depth1[0] + u) * x1.col(0) - (depth1[1] + u) * x1.col(1);
					Eigen::Vector3d u2 = (depth1[0] + u) * x1.col(0) - (depth1[2] + u) * x1.col(2);
					Eigen::Matrix3d X;
					X << u1, u2, u1.cross(u2);
					X = X.inverse().eval();

					Eigen::Matrix3d rot = Y * X;

					Eigen::Vector3d trans1 = (depth1[0] + u) * rot * x1.col(0);
					Eigen::Vector3d trans2 = s * (depth2[0] + v) * x2.col(0);
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

				return true;
			}
		}
	}
}