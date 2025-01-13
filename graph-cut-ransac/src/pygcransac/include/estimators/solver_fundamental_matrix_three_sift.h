// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
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
			class FundamentalMatrixThreeSIFTSolver : public SolverEngine
			{
			public:
				FundamentalMatrixThreeSIFTSolver()
				{
				}

				~FundamentalMatrixThreeSIFTSolver()
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
					return "3SIFTF";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 15;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sampleNumber_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			protected:
				Eigen::MatrixXcd solver_6pt_onefocal(const Eigen::VectorXd &data_) const;
			};

			OLGA_INLINE bool FundamentalMatrixThreeSIFTSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{				
				Eigen::MatrixXd coefficients(6, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double t0, t1, t2;
				int i, n;

				// Form a linear system: i-th row of A(=a) represents
				// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
				size_t row = 0;
				for (i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					const double
						u1 = data_ptr[offset],
						v1 = data_ptr[offset + 1],
						u2 = data_ptr[offset + 2],
						v2 = data_ptr[offset + 3],
						q1 = data_ptr[offset + 4],
						q2 = data_ptr[offset + 5],
						o1 = data_ptr[offset + 6],
						o2 = data_ptr[offset + 7];
    
					const double 
						s1 = sin(o1),
						c1 = cos(o1),
						s2 = sin(o2),
						c2 = cos(o2),
						q = q2 / q1;
						
					coefficients(row, 0) = u2 * u1;
					coefficients(row, 1) = u2 * v1;
					coefficients(row, 2) = u2;
					coefficients(row, 3) = v2 * u1;
					coefficients(row, 4) = v2 * v1;
					coefficients(row, 5) = v2;
					coefficients(row, 6) = u1;
					coefficients(row, 7) = v1;
					coefficients(row, 8) = 1;
					++row;

					coefficients(row, 0) = s2*s2 * q2 * u1 - c1 * c2 * q1 * u2 - q2*u1;
					coefficients(row, 1) = s2*s2 * q2 * v1 - s1 * c2 * q1 * u2 - q2*v1;
					coefficients(row, 2) = s2*s2 * q2 -q2;
					coefficients(row, 3) = -c2 * s2 * q2 * u1 - c1*c2*q1*v2;
					coefficients(row, 4) = -c2 * s2 * q2 * v1 - s1*c2*q1*v2;
					coefficients(row, 5) = -c2*s2*q2;
					coefficients(row, 6) = -c1*c2*q1;
					coefficients(row, 7) = -s1*c2*q1;
					coefficients(row, 8) = 0;
					++row;
				}

				Eigen::Matrix<double, 9, 1> f0, f1, f2;

				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
				if (lu.dimensionOfKernel() != 3) 
					return false;

				const Eigen::Matrix<double, 9, 3> null_space = 
					lu.kernel();

				f0 = null_space.col(0);
				f1 = null_space.col(1);
				f2 = null_space.col(2);




				Eigen::Matrix<double, 27, 1> data2;
				data2 << f0, f1, f2;

				Eigen::MatrixXcd sols = solver_6pt_onefocal(data2);

				
				for (size_t k = 0; k < 15; ++k)
				{
					if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() ||
						sols(1, k).imag() > std::numeric_limits<double>::epsilon())
						continue;

					Eigen::MatrixXd M = sols(0, k).real() * f0 + sols(1, k).real() * f1 + f2;

					double focal = 1.0 / sqrt(sols(2, k).real());
					// double focal = 1.0;
					double fm8 = focal/M(8);

					FundamentalMatrix model;
					model.descriptor << M(0)*fm8, M(1)*fm8, M(2)*fm8,
										M(3)*fm8, M(4)*fm8, M(5)*fm8,
										M(6)*fm8, M(7)*fm8, focal;

					models_.push_back(model);

					// FundamentalMatrix model;
					// model.descriptor << M(0), M(1), M(2),
					// 					M(3), M(4), M(5),
					// 					M(6), M(7), M(8);

					// // printf("%.3f", f1[8]);

					// models_.push_back(model);


				}


				return true;
			}

			Eigen::MatrixXcd FundamentalMatrixThreeSIFTSolver::solver_6pt_onefocal(const Eigen::VectorXd &data_) const
			{
				using namespace Eigen;
				// Action =  y
				// Quotient ring basis (V) = 1,x,x^2,x^2*y,x*y,x*y^2,y,y^2,y^3,
				// Available monomials (RR*V) = x^2*y^2,x*y^3,y^4,1,x,x^2,x^2*y,x*y,x*y^2,y,y^2,y^3,

				const double *d = data_.data();
				VectorXd coeffs(280);
				coeffs[0] = 2*d[2]*d[6]*d[8] - d[0]*std::pow(d[8],2);
				coeffs[1] = -std::pow(d[8],2)*d[9] + 2*d[6]*d[8]*d[11] + 2*d[2]*d[8]*d[15] + 2*d[2]*d[6]*d[17] - 2*d[0]*d[8]*d[17];
				coeffs[2] = 2*d[8]*d[11]*d[15] - 2*d[8]*d[9]*d[17] + 2*d[6]*d[11]*d[17] + 2*d[2]*d[15]*d[17] - d[0]*std::pow(d[17],2);
				coeffs[3] = 2*d[11]*d[15]*d[17] - d[9]*std::pow(d[17],2);
				coeffs[4] = d[0]*std::pow(d[2],2) + 2*d[2]*d[3]*d[5] - d[0]*std::pow(d[5],2) + d[0]*std::pow(d[6],2) + 2*d[1]*d[6]*d[7] - d[0]*std::pow(d[7],2);
				coeffs[5] = std::pow(d[2],2)*d[9] - std::pow(d[5],2)*d[9] + std::pow(d[6],2)*d[9] - std::pow(d[7],2)*d[9] + 2*d[6]*d[7]*d[10] + 2*d[0]*d[2]*d[11] + 2*d[3]*d[5]*d[11] + 2*d[2]*d[5]*d[12] + 2*d[2]*d[3]*d[14] - 2*d[0]*d[5]*d[14] + 2*d[0]*d[6]*d[15] + 2*d[1]*d[7]*d[15] + 2*d[1]*d[6]*d[16] - 2*d[0]*d[7]*d[16];
				coeffs[6] = 2*d[2]*d[9]*d[11] + d[0]*std::pow(d[11],2) + 2*d[5]*d[11]*d[12] - 2*d[5]*d[9]*d[14] + 2*d[3]*d[11]*d[14] + 2*d[2]*d[12]*d[14] - d[0]*std::pow(d[14],2) + 2*d[6]*d[9]*d[15] + 2*d[7]*d[10]*d[15] + d[0]*std::pow(d[15],2) - 2*d[7]*d[9]*d[16] + 2*d[6]*d[10]*d[16] + 2*d[1]*d[15]*d[16] - d[0]*std::pow(d[16],2);
				coeffs[7] = d[9]*std::pow(d[11],2) + 2*d[11]*d[12]*d[14] - d[9]*std::pow(d[14],2) + d[9]*std::pow(d[15],2) + 2*d[10]*d[15]*d[16] - d[9]*std::pow(d[16],2);
				coeffs[8] = -std::pow(d[8],2)*d[18] + 2*d[6]*d[8]*d[20] + 2*d[2]*d[8]*d[24] + 2*d[2]*d[6]*d[26] - 2*d[0]*d[8]*d[26];
				coeffs[9] = -2*d[8]*d[17]*d[18] + 2*d[8]*d[15]*d[20] + 2*d[6]*d[17]*d[20] + 2*d[8]*d[11]*d[24] + 2*d[2]*d[17]*d[24] - 2*d[8]*d[9]*d[26] + 2*d[6]*d[11]*d[26] + 2*d[2]*d[15]*d[26] - 2*d[0]*d[17]*d[26];
				coeffs[10] = -std::pow(d[17],2)*d[18] + 2*d[15]*d[17]*d[20] + 2*d[11]*d[17]*d[24] + 2*d[11]*d[15]*d[26] - 2*d[9]*d[17]*d[26];
				coeffs[11] = std::pow(d[0],3) + d[0]*std::pow(d[1],2) + d[0]*std::pow(d[3],2) + 2*d[1]*d[3]*d[4] - d[0]*std::pow(d[4],2);
				coeffs[12] = 3*std::pow(d[0],2)*d[9] + std::pow(d[1],2)*d[9] + std::pow(d[3],2)*d[9] - std::pow(d[4],2)*d[9] + 2*d[0]*d[1]*d[10] + 2*d[3]*d[4]*d[10] + 2*d[0]*d[3]*d[12] + 2*d[1]*d[4]*d[12] + 2*d[1]*d[3]*d[13] - 2*d[0]*d[4]*d[13];
				coeffs[13] = 3*d[0]*std::pow(d[9],2) + 2*d[1]*d[9]*d[10] + d[0]*std::pow(d[10],2) + 2*d[3]*d[9]*d[12] + 2*d[4]*d[10]*d[12] + d[0]*std::pow(d[12],2) - 2*d[4]*d[9]*d[13] + 2*d[3]*d[10]*d[13] + 2*d[1]*d[12]*d[13] - d[0]*std::pow(d[13],2);
				coeffs[14] = std::pow(d[9],3) + d[9]*std::pow(d[10],2) + d[9]*std::pow(d[12],2) + 2*d[10]*d[12]*d[13] - d[9]*std::pow(d[13],2);
				coeffs[15] = std::pow(d[2],2)*d[18] - std::pow(d[5],2)*d[18] + std::pow(d[6],2)*d[18] - std::pow(d[7],2)*d[18] + 2*d[6]*d[7]*d[19] + 2*d[0]*d[2]*d[20] + 2*d[3]*d[5]*d[20] + 2*d[2]*d[5]*d[21] + 2*d[2]*d[3]*d[23] - 2*d[0]*d[5]*d[23] + 2*d[0]*d[6]*d[24] + 2*d[1]*d[7]*d[24] + 2*d[1]*d[6]*d[25] - 2*d[0]*d[7]*d[25];
				coeffs[16] = 2*d[2]*d[11]*d[18] - 2*d[5]*d[14]*d[18] + 2*d[6]*d[15]*d[18] - 2*d[7]*d[16]*d[18] + 2*d[7]*d[15]*d[19] + 2*d[6]*d[16]*d[19] + 2*d[2]*d[9]*d[20] + 2*d[0]*d[11]*d[20] + 2*d[5]*d[12]*d[20] + 2*d[3]*d[14]*d[20] + 2*d[5]*d[11]*d[21] + 2*d[2]*d[14]*d[21] - 2*d[5]*d[9]*d[23] + 2*d[3]*d[11]*d[23] + 2*d[2]*d[12]*d[23] - 2*d[0]*d[14]*d[23] + 2*d[6]*d[9]*d[24] + 2*d[7]*d[10]*d[24] + 2*d[0]*d[15]*d[24] + 2*d[1]*d[16]*d[24] - 2*d[7]*d[9]*d[25] + 2*d[6]*d[10]*d[25] + 2*d[1]*d[15]*d[25] - 2*d[0]*d[16]*d[25];
				coeffs[17] = std::pow(d[11],2)*d[18] - std::pow(d[14],2)*d[18] + std::pow(d[15],2)*d[18] - std::pow(d[16],2)*d[18] + 2*d[15]*d[16]*d[19] + 2*d[9]*d[11]*d[20] + 2*d[12]*d[14]*d[20] + 2*d[11]*d[14]*d[21] + 2*d[11]*d[12]*d[23] - 2*d[9]*d[14]*d[23] + 2*d[9]*d[15]*d[24] + 2*d[10]*d[16]*d[24] + 2*d[10]*d[15]*d[25] - 2*d[9]*d[16]*d[25];
				coeffs[18] = 2*d[8]*d[20]*d[24] - 2*d[8]*d[18]*d[26] + 2*d[6]*d[20]*d[26] + 2*d[2]*d[24]*d[26] - d[0]*std::pow(d[26],2);
				coeffs[19] = 2*d[17]*d[20]*d[24] - 2*d[17]*d[18]*d[26] + 2*d[15]*d[20]*d[26] + 2*d[11]*d[24]*d[26] - d[9]*std::pow(d[26],2);
				coeffs[20] = 3*std::pow(d[0],2)*d[18] + std::pow(d[1],2)*d[18] + std::pow(d[3],2)*d[18] - std::pow(d[4],2)*d[18] + 2*d[0]*d[1]*d[19] + 2*d[3]*d[4]*d[19] + 2*d[0]*d[3]*d[21] + 2*d[1]*d[4]*d[21] + 2*d[1]*d[3]*d[22] - 2*d[0]*d[4]*d[22];
				coeffs[21] = 6*d[0]*d[9]*d[18] + 2*d[1]*d[10]*d[18] + 2*d[3]*d[12]*d[18] - 2*d[4]*d[13]*d[18] + 2*d[1]*d[9]*d[19] + 2*d[0]*d[10]*d[19] + 2*d[4]*d[12]*d[19] + 2*d[3]*d[13]*d[19] + 2*d[3]*d[9]*d[21] + 2*d[4]*d[10]*d[21] + 2*d[0]*d[12]*d[21] + 2*d[1]*d[13]*d[21] - 2*d[4]*d[9]*d[22] + 2*d[3]*d[10]*d[22] + 2*d[1]*d[12]*d[22] - 2*d[0]*d[13]*d[22];
				coeffs[22] = 3*std::pow(d[9],2)*d[18] + std::pow(d[10],2)*d[18] + std::pow(d[12],2)*d[18] - std::pow(d[13],2)*d[18] + 2*d[9]*d[10]*d[19] + 2*d[12]*d[13]*d[19] + 2*d[9]*d[12]*d[21] + 2*d[10]*d[13]*d[21] + 2*d[10]*d[12]*d[22] - 2*d[9]*d[13]*d[22];
				coeffs[23] = 2*d[2]*d[18]*d[20] + d[0]*std::pow(d[20],2) + 2*d[5]*d[20]*d[21] - 2*d[5]*d[18]*d[23] + 2*d[3]*d[20]*d[23] + 2*d[2]*d[21]*d[23] - d[0]*std::pow(d[23],2) + 2*d[6]*d[18]*d[24] + 2*d[7]*d[19]*d[24] + d[0]*std::pow(d[24],2) - 2*d[7]*d[18]*d[25] + 2*d[6]*d[19]*d[25] + 2*d[1]*d[24]*d[25] - d[0]*std::pow(d[25],2);
				coeffs[24] = 2*d[11]*d[18]*d[20] + d[9]*std::pow(d[20],2) + 2*d[14]*d[20]*d[21] - 2*d[14]*d[18]*d[23] + 2*d[12]*d[20]*d[23] + 2*d[11]*d[21]*d[23] - d[9]*std::pow(d[23],2) + 2*d[15]*d[18]*d[24] + 2*d[16]*d[19]*d[24] + d[9]*std::pow(d[24],2) - 2*d[16]*d[18]*d[25] + 2*d[15]*d[19]*d[25] + 2*d[10]*d[24]*d[25] - d[9]*std::pow(d[25],2);
				coeffs[25] = 2*d[20]*d[24]*d[26] - d[18]*std::pow(d[26],2);
				coeffs[26] = 3*d[0]*std::pow(d[18],2) + 2*d[1]*d[18]*d[19] + d[0]*std::pow(d[19],2) + 2*d[3]*d[18]*d[21] + 2*d[4]*d[19]*d[21] + d[0]*std::pow(d[21],2) - 2*d[4]*d[18]*d[22] + 2*d[3]*d[19]*d[22] + 2*d[1]*d[21]*d[22] - d[0]*std::pow(d[22],2);
				coeffs[27] = 3*d[9]*std::pow(d[18],2) + 2*d[10]*d[18]*d[19] + d[9]*std::pow(d[19],2) + 2*d[12]*d[18]*d[21] + 2*d[13]*d[19]*d[21] + d[9]*std::pow(d[21],2) - 2*d[13]*d[18]*d[22] + 2*d[12]*d[19]*d[22] + 2*d[10]*d[21]*d[22] - d[9]*std::pow(d[22],2);
				coeffs[28] = d[18]*std::pow(d[20],2) + 2*d[20]*d[21]*d[23] - d[18]*std::pow(d[23],2) + d[18]*std::pow(d[24],2) + 2*d[19]*d[24]*d[25] - d[18]*std::pow(d[25],2);
				coeffs[29] = std::pow(d[18],3) + d[18]*std::pow(d[19],2) + d[18]*std::pow(d[21],2) + 2*d[19]*d[21]*d[22] - d[18]*std::pow(d[22],2);
				coeffs[30] = 2*d[2]*d[7]*d[8] - d[1]*std::pow(d[8],2);
				coeffs[31] = -std::pow(d[8],2)*d[10] + 2*d[7]*d[8]*d[11] + 2*d[2]*d[8]*d[16] + 2*d[2]*d[7]*d[17] - 2*d[1]*d[8]*d[17];
				coeffs[32] = 2*d[8]*d[11]*d[16] - 2*d[8]*d[10]*d[17] + 2*d[7]*d[11]*d[17] + 2*d[2]*d[16]*d[17] - d[1]*std::pow(d[17],2);
				coeffs[33] = 2*d[11]*d[16]*d[17] - d[10]*std::pow(d[17],2);
				coeffs[34] = d[1]*std::pow(d[2],2) + 2*d[2]*d[4]*d[5] - d[1]*std::pow(d[5],2) - d[1]*std::pow(d[6],2) + 2*d[0]*d[6]*d[7] + d[1]*std::pow(d[7],2);
				coeffs[35] = 2*d[6]*d[7]*d[9] + std::pow(d[2],2)*d[10] - std::pow(d[5],2)*d[10] - std::pow(d[6],2)*d[10] + std::pow(d[7],2)*d[10] + 2*d[1]*d[2]*d[11] + 2*d[4]*d[5]*d[11] + 2*d[2]*d[5]*d[13] + 2*d[2]*d[4]*d[14] - 2*d[1]*d[5]*d[14] - 2*d[1]*d[6]*d[15] + 2*d[0]*d[7]*d[15] + 2*d[0]*d[6]*d[16] + 2*d[1]*d[7]*d[16];
				coeffs[36] = 2*d[2]*d[10]*d[11] + d[1]*std::pow(d[11],2) + 2*d[5]*d[11]*d[13] - 2*d[5]*d[10]*d[14] + 2*d[4]*d[11]*d[14] + 2*d[2]*d[13]*d[14] - d[1]*std::pow(d[14],2) + 2*d[7]*d[9]*d[15] - 2*d[6]*d[10]*d[15] - d[1]*std::pow(d[15],2) + 2*d[6]*d[9]*d[16] + 2*d[7]*d[10]*d[16] + 2*d[0]*d[15]*d[16] + d[1]*std::pow(d[16],2);
				coeffs[37] = d[10]*std::pow(d[11],2) + 2*d[11]*d[13]*d[14] - d[10]*std::pow(d[14],2) - d[10]*std::pow(d[15],2) + 2*d[9]*d[15]*d[16] + d[10]*std::pow(d[16],2);
				coeffs[38] = -std::pow(d[8],2)*d[19] + 2*d[7]*d[8]*d[20] + 2*d[2]*d[8]*d[25] + 2*d[2]*d[7]*d[26] - 2*d[1]*d[8]*d[26];
				coeffs[39] = -2*d[8]*d[17]*d[19] + 2*d[8]*d[16]*d[20] + 2*d[7]*d[17]*d[20] + 2*d[8]*d[11]*d[25] + 2*d[2]*d[17]*d[25] - 2*d[8]*d[10]*d[26] + 2*d[7]*d[11]*d[26] + 2*d[2]*d[16]*d[26] - 2*d[1]*d[17]*d[26];
				coeffs[40] = -std::pow(d[17],2)*d[19] + 2*d[16]*d[17]*d[20] + 2*d[11]*d[17]*d[25] + 2*d[11]*d[16]*d[26] - 2*d[10]*d[17]*d[26];
				coeffs[41] = std::pow(d[0],2)*d[1] + std::pow(d[1],3) - d[1]*std::pow(d[3],2) + 2*d[0]*d[3]*d[4] + d[1]*std::pow(d[4],2);
				coeffs[42] = 2*d[0]*d[1]*d[9] + 2*d[3]*d[4]*d[9] + std::pow(d[0],2)*d[10] + 3*std::pow(d[1],2)*d[10] - std::pow(d[3],2)*d[10] + std::pow(d[4],2)*d[10] - 2*d[1]*d[3]*d[12] + 2*d[0]*d[4]*d[12] + 2*d[0]*d[3]*d[13] + 2*d[1]*d[4]*d[13];
				coeffs[43] = d[1]*std::pow(d[9],2) + 2*d[0]*d[9]*d[10] + 3*d[1]*std::pow(d[10],2) + 2*d[4]*d[9]*d[12] - 2*d[3]*d[10]*d[12] - d[1]*std::pow(d[12],2) + 2*d[3]*d[9]*d[13] + 2*d[4]*d[10]*d[13] + 2*d[0]*d[12]*d[13] + d[1]*std::pow(d[13],2);
				coeffs[44] = std::pow(d[9],2)*d[10] + std::pow(d[10],3) - d[10]*std::pow(d[12],2) + 2*d[9]*d[12]*d[13] + d[10]*std::pow(d[13],2);
				coeffs[45] = 2*d[6]*d[7]*d[18] + std::pow(d[2],2)*d[19] - std::pow(d[5],2)*d[19] - std::pow(d[6],2)*d[19] + std::pow(d[7],2)*d[19] + 2*d[1]*d[2]*d[20] + 2*d[4]*d[5]*d[20] + 2*d[2]*d[5]*d[22] + 2*d[2]*d[4]*d[23] - 2*d[1]*d[5]*d[23] - 2*d[1]*d[6]*d[24] + 2*d[0]*d[7]*d[24] + 2*d[0]*d[6]*d[25] + 2*d[1]*d[7]*d[25];
				coeffs[46] = 2*d[7]*d[15]*d[18] + 2*d[6]*d[16]*d[18] + 2*d[2]*d[11]*d[19] - 2*d[5]*d[14]*d[19] - 2*d[6]*d[15]*d[19] + 2*d[7]*d[16]*d[19] + 2*d[2]*d[10]*d[20] + 2*d[1]*d[11]*d[20] + 2*d[5]*d[13]*d[20] + 2*d[4]*d[14]*d[20] + 2*d[5]*d[11]*d[22] + 2*d[2]*d[14]*d[22] - 2*d[5]*d[10]*d[23] + 2*d[4]*d[11]*d[23] + 2*d[2]*d[13]*d[23] - 2*d[1]*d[14]*d[23] + 2*d[7]*d[9]*d[24] - 2*d[6]*d[10]*d[24] - 2*d[1]*d[15]*d[24] + 2*d[0]*d[16]*d[24] + 2*d[6]*d[9]*d[25] + 2*d[7]*d[10]*d[25] + 2*d[0]*d[15]*d[25] + 2*d[1]*d[16]*d[25];
				coeffs[47] = 2*d[15]*d[16]*d[18] + std::pow(d[11],2)*d[19] - std::pow(d[14],2)*d[19] - std::pow(d[15],2)*d[19] + std::pow(d[16],2)*d[19] + 2*d[10]*d[11]*d[20] + 2*d[13]*d[14]*d[20] + 2*d[11]*d[14]*d[22] + 2*d[11]*d[13]*d[23] - 2*d[10]*d[14]*d[23] - 2*d[10]*d[15]*d[24] + 2*d[9]*d[16]*d[24] + 2*d[9]*d[15]*d[25] + 2*d[10]*d[16]*d[25];
				coeffs[48] = 2*d[8]*d[20]*d[25] - 2*d[8]*d[19]*d[26] + 2*d[7]*d[20]*d[26] + 2*d[2]*d[25]*d[26] - d[1]*std::pow(d[26],2);
				coeffs[49] = 2*d[17]*d[20]*d[25] - 2*d[17]*d[19]*d[26] + 2*d[16]*d[20]*d[26] + 2*d[11]*d[25]*d[26] - d[10]*std::pow(d[26],2);
				coeffs[50] = 2*d[0]*d[1]*d[18] + 2*d[3]*d[4]*d[18] + std::pow(d[0],2)*d[19] + 3*std::pow(d[1],2)*d[19] - std::pow(d[3],2)*d[19] + std::pow(d[4],2)*d[19] - 2*d[1]*d[3]*d[21] + 2*d[0]*d[4]*d[21] + 2*d[0]*d[3]*d[22] + 2*d[1]*d[4]*d[22];
				coeffs[51] = 2*d[1]*d[9]*d[18] + 2*d[0]*d[10]*d[18] + 2*d[4]*d[12]*d[18] + 2*d[3]*d[13]*d[18] + 2*d[0]*d[9]*d[19] + 6*d[1]*d[10]*d[19] - 2*d[3]*d[12]*d[19] + 2*d[4]*d[13]*d[19] + 2*d[4]*d[9]*d[21] - 2*d[3]*d[10]*d[21] - 2*d[1]*d[12]*d[21] + 2*d[0]*d[13]*d[21] + 2*d[3]*d[9]*d[22] + 2*d[4]*d[10]*d[22] + 2*d[0]*d[12]*d[22] + 2*d[1]*d[13]*d[22];
				coeffs[52] = 2*d[9]*d[10]*d[18] + 2*d[12]*d[13]*d[18] + std::pow(d[9],2)*d[19] + 3*std::pow(d[10],2)*d[19] - std::pow(d[12],2)*d[19] + std::pow(d[13],2)*d[19] - 2*d[10]*d[12]*d[21] + 2*d[9]*d[13]*d[21] + 2*d[9]*d[12]*d[22] + 2*d[10]*d[13]*d[22];
				coeffs[53] = 2*d[2]*d[19]*d[20] + d[1]*std::pow(d[20],2) + 2*d[5]*d[20]*d[22] - 2*d[5]*d[19]*d[23] + 2*d[4]*d[20]*d[23] + 2*d[2]*d[22]*d[23] - d[1]*std::pow(d[23],2) + 2*d[7]*d[18]*d[24] - 2*d[6]*d[19]*d[24] - d[1]*std::pow(d[24],2) + 2*d[6]*d[18]*d[25] + 2*d[7]*d[19]*d[25] + 2*d[0]*d[24]*d[25] + d[1]*std::pow(d[25],2);
				coeffs[54] = 2*d[11]*d[19]*d[20] + d[10]*std::pow(d[20],2) + 2*d[14]*d[20]*d[22] - 2*d[14]*d[19]*d[23] + 2*d[13]*d[20]*d[23] + 2*d[11]*d[22]*d[23] - d[10]*std::pow(d[23],2) + 2*d[16]*d[18]*d[24] - 2*d[15]*d[19]*d[24] - d[10]*std::pow(d[24],2) + 2*d[15]*d[18]*d[25] + 2*d[16]*d[19]*d[25] + 2*d[9]*d[24]*d[25] + d[10]*std::pow(d[25],2);
				coeffs[55] = 2*d[20]*d[25]*d[26] - d[19]*std::pow(d[26],2);
				coeffs[56] = d[1]*std::pow(d[18],2) + 2*d[0]*d[18]*d[19] + 3*d[1]*std::pow(d[19],2) + 2*d[4]*d[18]*d[21] - 2*d[3]*d[19]*d[21] - d[1]*std::pow(d[21],2) + 2*d[3]*d[18]*d[22] + 2*d[4]*d[19]*d[22] + 2*d[0]*d[21]*d[22] + d[1]*std::pow(d[22],2);
				coeffs[57] = d[10]*std::pow(d[18],2) + 2*d[9]*d[18]*d[19] + 3*d[10]*std::pow(d[19],2) + 2*d[13]*d[18]*d[21] - 2*d[12]*d[19]*d[21] - d[10]*std::pow(d[21],2) + 2*d[12]*d[18]*d[22] + 2*d[13]*d[19]*d[22] + 2*d[9]*d[21]*d[22] + d[10]*std::pow(d[22],2);
				coeffs[58] = d[19]*std::pow(d[20],2) + 2*d[20]*d[22]*d[23] - d[19]*std::pow(d[23],2) - d[19]*std::pow(d[24],2) + 2*d[18]*d[24]*d[25] + d[19]*std::pow(d[25],2);
				coeffs[59] = std::pow(d[18],2)*d[19] + std::pow(d[19],3) - d[19]*std::pow(d[21],2) + 2*d[18]*d[21]*d[22] + d[19]*std::pow(d[22],2);
				coeffs[60] = d[2]*std::pow(d[8],2);
				coeffs[61] = std::pow(d[8],2)*d[11] + 2*d[2]*d[8]*d[17];
				coeffs[62] = 2*d[8]*d[11]*d[17] + d[2]*std::pow(d[17],2);
				coeffs[63] = d[11]*std::pow(d[17],2);
				coeffs[64] = std::pow(d[2],3) + d[2]*std::pow(d[5],2) - d[2]*std::pow(d[6],2) - d[2]*std::pow(d[7],2) + 2*d[0]*d[6]*d[8] + 2*d[1]*d[7]*d[8];
				coeffs[65] = 2*d[6]*d[8]*d[9] + 2*d[7]*d[8]*d[10] + 3*std::pow(d[2],2)*d[11] + std::pow(d[5],2)*d[11] - std::pow(d[6],2)*d[11] - std::pow(d[7],2)*d[11] + 2*d[2]*d[5]*d[14] - 2*d[2]*d[6]*d[15] + 2*d[0]*d[8]*d[15] - 2*d[2]*d[7]*d[16] + 2*d[1]*d[8]*d[16] + 2*d[0]*d[6]*d[17] + 2*d[1]*d[7]*d[17];
				coeffs[66] = 3*d[2]*std::pow(d[11],2) + 2*d[5]*d[11]*d[14] + d[2]*std::pow(d[14],2) + 2*d[8]*d[9]*d[15] - 2*d[6]*d[11]*d[15] - d[2]*std::pow(d[15],2) + 2*d[8]*d[10]*d[16] - 2*d[7]*d[11]*d[16] - d[2]*std::pow(d[16],2) + 2*d[6]*d[9]*d[17] + 2*d[7]*d[10]*d[17] + 2*d[0]*d[15]*d[17] + 2*d[1]*d[16]*d[17];
				coeffs[67] = std::pow(d[11],3) + d[11]*std::pow(d[14],2) - d[11]*std::pow(d[15],2) - d[11]*std::pow(d[16],2) + 2*d[9]*d[15]*d[17] + 2*d[10]*d[16]*d[17];
				coeffs[68] = std::pow(d[8],2)*d[20] + 2*d[2]*d[8]*d[26];
				coeffs[69] = 2*d[8]*d[17]*d[20] + 2*d[8]*d[11]*d[26] + 2*d[2]*d[17]*d[26];
				coeffs[70] = std::pow(d[17],2)*d[20] + 2*d[11]*d[17]*d[26];
				coeffs[71] = std::pow(d[0],2)*d[2] + std::pow(d[1],2)*d[2] - d[2]*std::pow(d[3],2) - d[2]*std::pow(d[4],2) + 2*d[0]*d[3]*d[5] + 2*d[1]*d[4]*d[5];
				coeffs[72] = 2*d[0]*d[2]*d[9] + 2*d[3]*d[5]*d[9] + 2*d[1]*d[2]*d[10] + 2*d[4]*d[5]*d[10] + std::pow(d[0],2)*d[11] + std::pow(d[1],2)*d[11] - std::pow(d[3],2)*d[11] - std::pow(d[4],2)*d[11] - 2*d[2]*d[3]*d[12] + 2*d[0]*d[5]*d[12] - 2*d[2]*d[4]*d[13] + 2*d[1]*d[5]*d[13] + 2*d[0]*d[3]*d[14] + 2*d[1]*d[4]*d[14];
				coeffs[73] = d[2]*std::pow(d[9],2) + d[2]*std::pow(d[10],2) + 2*d[0]*d[9]*d[11] + 2*d[1]*d[10]*d[11] + 2*d[5]*d[9]*d[12] - 2*d[3]*d[11]*d[12] - d[2]*std::pow(d[12],2) + 2*d[5]*d[10]*d[13] - 2*d[4]*d[11]*d[13] - d[2]*std::pow(d[13],2) + 2*d[3]*d[9]*d[14] + 2*d[4]*d[10]*d[14] + 2*d[0]*d[12]*d[14] + 2*d[1]*d[13]*d[14];
				coeffs[74] = std::pow(d[9],2)*d[11] + std::pow(d[10],2)*d[11] - d[11]*std::pow(d[12],2) - d[11]*std::pow(d[13],2) + 2*d[9]*d[12]*d[14] + 2*d[10]*d[13]*d[14];
				coeffs[75] = 2*d[6]*d[8]*d[18] + 2*d[7]*d[8]*d[19] + 3*std::pow(d[2],2)*d[20] + std::pow(d[5],2)*d[20] - std::pow(d[6],2)*d[20] - std::pow(d[7],2)*d[20] + 2*d[2]*d[5]*d[23] - 2*d[2]*d[6]*d[24] + 2*d[0]*d[8]*d[24] - 2*d[2]*d[7]*d[25] + 2*d[1]*d[8]*d[25] + 2*d[0]*d[6]*d[26] + 2*d[1]*d[7]*d[26];
				coeffs[76] = 2*d[8]*d[15]*d[18] + 2*d[6]*d[17]*d[18] + 2*d[8]*d[16]*d[19] + 2*d[7]*d[17]*d[19] + 6*d[2]*d[11]*d[20] + 2*d[5]*d[14]*d[20] - 2*d[6]*d[15]*d[20] - 2*d[7]*d[16]*d[20] + 2*d[5]*d[11]*d[23] + 2*d[2]*d[14]*d[23] + 2*d[8]*d[9]*d[24] - 2*d[6]*d[11]*d[24] - 2*d[2]*d[15]*d[24] + 2*d[0]*d[17]*d[24] + 2*d[8]*d[10]*d[25] - 2*d[7]*d[11]*d[25] - 2*d[2]*d[16]*d[25] + 2*d[1]*d[17]*d[25] + 2*d[6]*d[9]*d[26] + 2*d[7]*d[10]*d[26] + 2*d[0]*d[15]*d[26] + 2*d[1]*d[16]*d[26];
				coeffs[77] = 2*d[15]*d[17]*d[18] + 2*d[16]*d[17]*d[19] + 3*std::pow(d[11],2)*d[20] + std::pow(d[14],2)*d[20] - std::pow(d[15],2)*d[20] - std::pow(d[16],2)*d[20] + 2*d[11]*d[14]*d[23] - 2*d[11]*d[15]*d[24] + 2*d[9]*d[17]*d[24] - 2*d[11]*d[16]*d[25] + 2*d[10]*d[17]*d[25] + 2*d[9]*d[15]*d[26] + 2*d[10]*d[16]*d[26];
				coeffs[78] = 2*d[8]*d[20]*d[26] + d[2]*std::pow(d[26],2);
				coeffs[79] = 2*d[17]*d[20]*d[26] + d[11]*std::pow(d[26],2);
				coeffs[80] = 2*d[0]*d[2]*d[18] + 2*d[3]*d[5]*d[18] + 2*d[1]*d[2]*d[19] + 2*d[4]*d[5]*d[19] + std::pow(d[0],2)*d[20] + std::pow(d[1],2)*d[20] - std::pow(d[3],2)*d[20] - std::pow(d[4],2)*d[20] - 2*d[2]*d[3]*d[21] + 2*d[0]*d[5]*d[21] - 2*d[2]*d[4]*d[22] + 2*d[1]*d[5]*d[22] + 2*d[0]*d[3]*d[23] + 2*d[1]*d[4]*d[23];
				coeffs[81] = 2*d[2]*d[9]*d[18] + 2*d[0]*d[11]*d[18] + 2*d[5]*d[12]*d[18] + 2*d[3]*d[14]*d[18] + 2*d[2]*d[10]*d[19] + 2*d[1]*d[11]*d[19] + 2*d[5]*d[13]*d[19] + 2*d[4]*d[14]*d[19] + 2*d[0]*d[9]*d[20] + 2*d[1]*d[10]*d[20] - 2*d[3]*d[12]*d[20] - 2*d[4]*d[13]*d[20] + 2*d[5]*d[9]*d[21] - 2*d[3]*d[11]*d[21] - 2*d[2]*d[12]*d[21] + 2*d[0]*d[14]*d[21] + 2*d[5]*d[10]*d[22] - 2*d[4]*d[11]*d[22] - 2*d[2]*d[13]*d[22] + 2*d[1]*d[14]*d[22] + 2*d[3]*d[9]*d[23] + 2*d[4]*d[10]*d[23] + 2*d[0]*d[12]*d[23] + 2*d[1]*d[13]*d[23];
				coeffs[82] = 2*d[9]*d[11]*d[18] + 2*d[12]*d[14]*d[18] + 2*d[10]*d[11]*d[19] + 2*d[13]*d[14]*d[19] + std::pow(d[9],2)*d[20] + std::pow(d[10],2)*d[20] - std::pow(d[12],2)*d[20] - std::pow(d[13],2)*d[20] - 2*d[11]*d[12]*d[21] + 2*d[9]*d[14]*d[21] - 2*d[11]*d[13]*d[22] + 2*d[10]*d[14]*d[22] + 2*d[9]*d[12]*d[23] + 2*d[10]*d[13]*d[23];
				coeffs[83] = 3*d[2]*std::pow(d[20],2) + 2*d[5]*d[20]*d[23] + d[2]*std::pow(d[23],2) + 2*d[8]*d[18]*d[24] - 2*d[6]*d[20]*d[24] - d[2]*std::pow(d[24],2) + 2*d[8]*d[19]*d[25] - 2*d[7]*d[20]*d[25] - d[2]*std::pow(d[25],2) + 2*d[6]*d[18]*d[26] + 2*d[7]*d[19]*d[26] + 2*d[0]*d[24]*d[26] + 2*d[1]*d[25]*d[26];
				coeffs[84] = 3*d[11]*std::pow(d[20],2) + 2*d[14]*d[20]*d[23] + d[11]*std::pow(d[23],2) + 2*d[17]*d[18]*d[24] - 2*d[15]*d[20]*d[24] - d[11]*std::pow(d[24],2) + 2*d[17]*d[19]*d[25] - 2*d[16]*d[20]*d[25] - d[11]*std::pow(d[25],2) + 2*d[15]*d[18]*d[26] + 2*d[16]*d[19]*d[26] + 2*d[9]*d[24]*d[26] + 2*d[10]*d[25]*d[26];
				coeffs[85] = d[20]*std::pow(d[26],2);
				coeffs[86] = d[2]*std::pow(d[18],2) + d[2]*std::pow(d[19],2) + 2*d[0]*d[18]*d[20] + 2*d[1]*d[19]*d[20] + 2*d[5]*d[18]*d[21] - 2*d[3]*d[20]*d[21] - d[2]*std::pow(d[21],2) + 2*d[5]*d[19]*d[22] - 2*d[4]*d[20]*d[22] - d[2]*std::pow(d[22],2) + 2*d[3]*d[18]*d[23] + 2*d[4]*d[19]*d[23] + 2*d[0]*d[21]*d[23] + 2*d[1]*d[22]*d[23];
				coeffs[87] = d[11]*std::pow(d[18],2) + d[11]*std::pow(d[19],2) + 2*d[9]*d[18]*d[20] + 2*d[10]*d[19]*d[20] + 2*d[14]*d[18]*d[21] - 2*d[12]*d[20]*d[21] - d[11]*std::pow(d[21],2) + 2*d[14]*d[19]*d[22] - 2*d[13]*d[20]*d[22] - d[11]*std::pow(d[22],2) + 2*d[12]*d[18]*d[23] + 2*d[13]*d[19]*d[23] + 2*d[9]*d[21]*d[23] + 2*d[10]*d[22]*d[23];
				coeffs[88] = std::pow(d[20],3) + d[20]*std::pow(d[23],2) - d[20]*std::pow(d[24],2) - d[20]*std::pow(d[25],2) + 2*d[18]*d[24]*d[26] + 2*d[19]*d[25]*d[26];
				coeffs[89] = std::pow(d[18],2)*d[20] + std::pow(d[19],2)*d[20] - d[20]*std::pow(d[21],2) - d[20]*std::pow(d[22],2) + 2*d[18]*d[21]*d[23] + 2*d[19]*d[22]*d[23];
				coeffs[90] = 2*d[5]*d[6]*d[8] - d[3]*std::pow(d[8],2);
				coeffs[91] = -std::pow(d[8],2)*d[12] + 2*d[6]*d[8]*d[14] + 2*d[5]*d[8]*d[15] + 2*d[5]*d[6]*d[17] - 2*d[3]*d[8]*d[17];
				coeffs[92] = 2*d[8]*d[14]*d[15] - 2*d[8]*d[12]*d[17] + 2*d[6]*d[14]*d[17] + 2*d[5]*d[15]*d[17] - d[3]*std::pow(d[17],2);
				coeffs[93] = 2*d[14]*d[15]*d[17] - d[12]*std::pow(d[17],2);
				coeffs[94] = -std::pow(d[2],2)*d[3] + 2*d[0]*d[2]*d[5] + d[3]*std::pow(d[5],2) + d[3]*std::pow(d[6],2) + 2*d[4]*d[6]*d[7] - d[3]*std::pow(d[7],2);
				coeffs[95] = 2*d[2]*d[5]*d[9] - 2*d[2]*d[3]*d[11] + 2*d[0]*d[5]*d[11] - std::pow(d[2],2)*d[12] + std::pow(d[5],2)*d[12] + std::pow(d[6],2)*d[12] - std::pow(d[7],2)*d[12] + 2*d[6]*d[7]*d[13] + 2*d[0]*d[2]*d[14] + 2*d[3]*d[5]*d[14] + 2*d[3]*d[6]*d[15] + 2*d[4]*d[7]*d[15] + 2*d[4]*d[6]*d[16] - 2*d[3]*d[7]*d[16];
				coeffs[96] = 2*d[5]*d[9]*d[11] - d[3]*std::pow(d[11],2) - 2*d[2]*d[11]*d[12] + 2*d[2]*d[9]*d[14] + 2*d[0]*d[11]*d[14] + 2*d[5]*d[12]*d[14] + d[3]*std::pow(d[14],2) + 2*d[6]*d[12]*d[15] + 2*d[7]*d[13]*d[15] + d[3]*std::pow(d[15],2) - 2*d[7]*d[12]*d[16] + 2*d[6]*d[13]*d[16] + 2*d[4]*d[15]*d[16] - d[3]*std::pow(d[16],2);
				coeffs[97] = -std::pow(d[11],2)*d[12] + 2*d[9]*d[11]*d[14] + d[12]*std::pow(d[14],2) + d[12]*std::pow(d[15],2) + 2*d[13]*d[15]*d[16] - d[12]*std::pow(d[16],2);
				coeffs[98] = -std::pow(d[8],2)*d[21] + 2*d[6]*d[8]*d[23] + 2*d[5]*d[8]*d[24] + 2*d[5]*d[6]*d[26] - 2*d[3]*d[8]*d[26];
				coeffs[99] = -2*d[8]*d[17]*d[21] + 2*d[8]*d[15]*d[23] + 2*d[6]*d[17]*d[23] + 2*d[8]*d[14]*d[24] + 2*d[5]*d[17]*d[24] - 2*d[8]*d[12]*d[26] + 2*d[6]*d[14]*d[26] + 2*d[5]*d[15]*d[26] - 2*d[3]*d[17]*d[26];
				coeffs[100] = -std::pow(d[17],2)*d[21] + 2*d[15]*d[17]*d[23] + 2*d[14]*d[17]*d[24] + 2*d[14]*d[15]*d[26] - 2*d[12]*d[17]*d[26];
				coeffs[101] = std::pow(d[0],2)*d[3] - std::pow(d[1],2)*d[3] + std::pow(d[3],3) + 2*d[0]*d[1]*d[4] + d[3]*std::pow(d[4],2);
				coeffs[102] = 2*d[0]*d[3]*d[9] + 2*d[1]*d[4]*d[9] - 2*d[1]*d[3]*d[10] + 2*d[0]*d[4]*d[10] + std::pow(d[0],2)*d[12] - std::pow(d[1],2)*d[12] + 3*std::pow(d[3],2)*d[12] + std::pow(d[4],2)*d[12] + 2*d[0]*d[1]*d[13] + 2*d[3]*d[4]*d[13];
				coeffs[103] = d[3]*std::pow(d[9],2) + 2*d[4]*d[9]*d[10] - d[3]*std::pow(d[10],2) + 2*d[0]*d[9]*d[12] - 2*d[1]*d[10]*d[12] + 3*d[3]*std::pow(d[12],2) + 2*d[1]*d[9]*d[13] + 2*d[0]*d[10]*d[13] + 2*d[4]*d[12]*d[13] + d[3]*std::pow(d[13],2);
				coeffs[104] = std::pow(d[9],2)*d[12] - std::pow(d[10],2)*d[12] + std::pow(d[12],3) + 2*d[9]*d[10]*d[13] + d[12]*std::pow(d[13],2);
				coeffs[105] = 2*d[2]*d[5]*d[18] - 2*d[2]*d[3]*d[20] + 2*d[0]*d[5]*d[20] - std::pow(d[2],2)*d[21] + std::pow(d[5],2)*d[21] + std::pow(d[6],2)*d[21] - std::pow(d[7],2)*d[21] + 2*d[6]*d[7]*d[22] + 2*d[0]*d[2]*d[23] + 2*d[3]*d[5]*d[23] + 2*d[3]*d[6]*d[24] + 2*d[4]*d[7]*d[24] + 2*d[4]*d[6]*d[25] - 2*d[3]*d[7]*d[25];
				coeffs[106] = 2*d[5]*d[11]*d[18] + 2*d[2]*d[14]*d[18] + 2*d[5]*d[9]*d[20] - 2*d[3]*d[11]*d[20] - 2*d[2]*d[12]*d[20] + 2*d[0]*d[14]*d[20] - 2*d[2]*d[11]*d[21] + 2*d[5]*d[14]*d[21] + 2*d[6]*d[15]*d[21] - 2*d[7]*d[16]*d[21] + 2*d[7]*d[15]*d[22] + 2*d[6]*d[16]*d[22] + 2*d[2]*d[9]*d[23] + 2*d[0]*d[11]*d[23] + 2*d[5]*d[12]*d[23] + 2*d[3]*d[14]*d[23] + 2*d[6]*d[12]*d[24] + 2*d[7]*d[13]*d[24] + 2*d[3]*d[15]*d[24] + 2*d[4]*d[16]*d[24] - 2*d[7]*d[12]*d[25] + 2*d[6]*d[13]*d[25] + 2*d[4]*d[15]*d[25] - 2*d[3]*d[16]*d[25];
				coeffs[107] = 2*d[11]*d[14]*d[18] - 2*d[11]*d[12]*d[20] + 2*d[9]*d[14]*d[20] - std::pow(d[11],2)*d[21] + std::pow(d[14],2)*d[21] + std::pow(d[15],2)*d[21] - std::pow(d[16],2)*d[21] + 2*d[15]*d[16]*d[22] + 2*d[9]*d[11]*d[23] + 2*d[12]*d[14]*d[23] + 2*d[12]*d[15]*d[24] + 2*d[13]*d[16]*d[24] + 2*d[13]*d[15]*d[25] - 2*d[12]*d[16]*d[25];
				coeffs[108] = 2*d[8]*d[23]*d[24] - 2*d[8]*d[21]*d[26] + 2*d[6]*d[23]*d[26] + 2*d[5]*d[24]*d[26] - d[3]*std::pow(d[26],2);
				coeffs[109] = 2*d[17]*d[23]*d[24] - 2*d[17]*d[21]*d[26] + 2*d[15]*d[23]*d[26] + 2*d[14]*d[24]*d[26] - d[12]*std::pow(d[26],2);
				coeffs[110] = 2*d[0]*d[3]*d[18] + 2*d[1]*d[4]*d[18] - 2*d[1]*d[3]*d[19] + 2*d[0]*d[4]*d[19] + std::pow(d[0],2)*d[21] - std::pow(d[1],2)*d[21] + 3*std::pow(d[3],2)*d[21] + std::pow(d[4],2)*d[21] + 2*d[0]*d[1]*d[22] + 2*d[3]*d[4]*d[22];
				coeffs[111] = 2*d[3]*d[9]*d[18] + 2*d[4]*d[10]*d[18] + 2*d[0]*d[12]*d[18] + 2*d[1]*d[13]*d[18] + 2*d[4]*d[9]*d[19] - 2*d[3]*d[10]*d[19] - 2*d[1]*d[12]*d[19] + 2*d[0]*d[13]*d[19] + 2*d[0]*d[9]*d[21] - 2*d[1]*d[10]*d[21] + 6*d[3]*d[12]*d[21] + 2*d[4]*d[13]*d[21] + 2*d[1]*d[9]*d[22] + 2*d[0]*d[10]*d[22] + 2*d[4]*d[12]*d[22] + 2*d[3]*d[13]*d[22];
				coeffs[112] = 2*d[9]*d[12]*d[18] + 2*d[10]*d[13]*d[18] - 2*d[10]*d[12]*d[19] + 2*d[9]*d[13]*d[19] + std::pow(d[9],2)*d[21] - std::pow(d[10],2)*d[21] + 3*std::pow(d[12],2)*d[21] + std::pow(d[13],2)*d[21] + 2*d[9]*d[10]*d[22] + 2*d[12]*d[13]*d[22];
				coeffs[113] = 2*d[5]*d[18]*d[20] - d[3]*std::pow(d[20],2) - 2*d[2]*d[20]*d[21] + 2*d[2]*d[18]*d[23] + 2*d[0]*d[20]*d[23] + 2*d[5]*d[21]*d[23] + d[3]*std::pow(d[23],2) + 2*d[6]*d[21]*d[24] + 2*d[7]*d[22]*d[24] + d[3]*std::pow(d[24],2) - 2*d[7]*d[21]*d[25] + 2*d[6]*d[22]*d[25] + 2*d[4]*d[24]*d[25] - d[3]*std::pow(d[25],2);
				coeffs[114] = 2*d[14]*d[18]*d[20] - d[12]*std::pow(d[20],2) - 2*d[11]*d[20]*d[21] + 2*d[11]*d[18]*d[23] + 2*d[9]*d[20]*d[23] + 2*d[14]*d[21]*d[23] + d[12]*std::pow(d[23],2) + 2*d[15]*d[21]*d[24] + 2*d[16]*d[22]*d[24] + d[12]*std::pow(d[24],2) - 2*d[16]*d[21]*d[25] + 2*d[15]*d[22]*d[25] + 2*d[13]*d[24]*d[25] - d[12]*std::pow(d[25],2);
				coeffs[115] = 2*d[23]*d[24]*d[26] - d[21]*std::pow(d[26],2);
				coeffs[116] = d[3]*std::pow(d[18],2) + 2*d[4]*d[18]*d[19] - d[3]*std::pow(d[19],2) + 2*d[0]*d[18]*d[21] - 2*d[1]*d[19]*d[21] + 3*d[3]*std::pow(d[21],2) + 2*d[1]*d[18]*d[22] + 2*d[0]*d[19]*d[22] + 2*d[4]*d[21]*d[22] + d[3]*std::pow(d[22],2);
				coeffs[117] = d[12]*std::pow(d[18],2) + 2*d[13]*d[18]*d[19] - d[12]*std::pow(d[19],2) + 2*d[9]*d[18]*d[21] - 2*d[10]*d[19]*d[21] + 3*d[12]*std::pow(d[21],2) + 2*d[10]*d[18]*d[22] + 2*d[9]*d[19]*d[22] + 2*d[13]*d[21]*d[22] + d[12]*std::pow(d[22],2);
				coeffs[118] = -std::pow(d[20],2)*d[21] + 2*d[18]*d[20]*d[23] + d[21]*std::pow(d[23],2) + d[21]*std::pow(d[24],2) + 2*d[22]*d[24]*d[25] - d[21]*std::pow(d[25],2);
				coeffs[119] = std::pow(d[18],2)*d[21] - std::pow(d[19],2)*d[21] + std::pow(d[21],3) + 2*d[18]*d[19]*d[22] + d[21]*std::pow(d[22],2);
				coeffs[120] = 2*d[5]*d[7]*d[8] - d[4]*std::pow(d[8],2);
				coeffs[121] = -std::pow(d[8],2)*d[13] + 2*d[7]*d[8]*d[14] + 2*d[5]*d[8]*d[16] + 2*d[5]*d[7]*d[17] - 2*d[4]*d[8]*d[17];
				coeffs[122] = 2*d[8]*d[14]*d[16] - 2*d[8]*d[13]*d[17] + 2*d[7]*d[14]*d[17] + 2*d[5]*d[16]*d[17] - d[4]*std::pow(d[17],2);
				coeffs[123] = 2*d[14]*d[16]*d[17] - d[13]*std::pow(d[17],2);
				coeffs[124] = -std::pow(d[2],2)*d[4] + 2*d[1]*d[2]*d[5] + d[4]*std::pow(d[5],2) - d[4]*std::pow(d[6],2) + 2*d[3]*d[6]*d[7] + d[4]*std::pow(d[7],2);
				coeffs[125] = 2*d[2]*d[5]*d[10] - 2*d[2]*d[4]*d[11] + 2*d[1]*d[5]*d[11] + 2*d[6]*d[7]*d[12] - std::pow(d[2],2)*d[13] + std::pow(d[5],2)*d[13] - std::pow(d[6],2)*d[13] + std::pow(d[7],2)*d[13] + 2*d[1]*d[2]*d[14] + 2*d[4]*d[5]*d[14] - 2*d[4]*d[6]*d[15] + 2*d[3]*d[7]*d[15] + 2*d[3]*d[6]*d[16] + 2*d[4]*d[7]*d[16];
				coeffs[126] = 2*d[5]*d[10]*d[11] - d[4]*std::pow(d[11],2) - 2*d[2]*d[11]*d[13] + 2*d[2]*d[10]*d[14] + 2*d[1]*d[11]*d[14] + 2*d[5]*d[13]*d[14] + d[4]*std::pow(d[14],2) + 2*d[7]*d[12]*d[15] - 2*d[6]*d[13]*d[15] - d[4]*std::pow(d[15],2) + 2*d[6]*d[12]*d[16] + 2*d[7]*d[13]*d[16] + 2*d[3]*d[15]*d[16] + d[4]*std::pow(d[16],2);
				coeffs[127] = -std::pow(d[11],2)*d[13] + 2*d[10]*d[11]*d[14] + d[13]*std::pow(d[14],2) - d[13]*std::pow(d[15],2) + 2*d[12]*d[15]*d[16] + d[13]*std::pow(d[16],2);
				coeffs[128] = -std::pow(d[8],2)*d[22] + 2*d[7]*d[8]*d[23] + 2*d[5]*d[8]*d[25] + 2*d[5]*d[7]*d[26] - 2*d[4]*d[8]*d[26];
				coeffs[129] = -2*d[8]*d[17]*d[22] + 2*d[8]*d[16]*d[23] + 2*d[7]*d[17]*d[23] + 2*d[8]*d[14]*d[25] + 2*d[5]*d[17]*d[25] - 2*d[8]*d[13]*d[26] + 2*d[7]*d[14]*d[26] + 2*d[5]*d[16]*d[26] - 2*d[4]*d[17]*d[26];
				coeffs[130] = -std::pow(d[17],2)*d[22] + 2*d[16]*d[17]*d[23] + 2*d[14]*d[17]*d[25] + 2*d[14]*d[16]*d[26] - 2*d[13]*d[17]*d[26];
				coeffs[131] = 2*d[0]*d[1]*d[3] - std::pow(d[0],2)*d[4] + std::pow(d[1],2)*d[4] + std::pow(d[3],2)*d[4] + std::pow(d[4],3);
				coeffs[132] = 2*d[1]*d[3]*d[9] - 2*d[0]*d[4]*d[9] + 2*d[0]*d[3]*d[10] + 2*d[1]*d[4]*d[10] + 2*d[0]*d[1]*d[12] + 2*d[3]*d[4]*d[12] - std::pow(d[0],2)*d[13] + std::pow(d[1],2)*d[13] + std::pow(d[3],2)*d[13] + 3*std::pow(d[4],2)*d[13];
				coeffs[133] = -d[4]*std::pow(d[9],2) + 2*d[3]*d[9]*d[10] + d[4]*std::pow(d[10],2) + 2*d[1]*d[9]*d[12] + 2*d[0]*d[10]*d[12] + d[4]*std::pow(d[12],2) - 2*d[0]*d[9]*d[13] + 2*d[1]*d[10]*d[13] + 2*d[3]*d[12]*d[13] + 3*d[4]*std::pow(d[13],2);
				coeffs[134] = 2*d[9]*d[10]*d[12] - std::pow(d[9],2)*d[13] + std::pow(d[10],2)*d[13] + std::pow(d[12],2)*d[13] + std::pow(d[13],3);
				coeffs[135] = 2*d[2]*d[5]*d[19] - 2*d[2]*d[4]*d[20] + 2*d[1]*d[5]*d[20] + 2*d[6]*d[7]*d[21] - std::pow(d[2],2)*d[22] + std::pow(d[5],2)*d[22] - std::pow(d[6],2)*d[22] + std::pow(d[7],2)*d[22] + 2*d[1]*d[2]*d[23] + 2*d[4]*d[5]*d[23] - 2*d[4]*d[6]*d[24] + 2*d[3]*d[7]*d[24] + 2*d[3]*d[6]*d[25] + 2*d[4]*d[7]*d[25];
				coeffs[136] = 2*d[5]*d[11]*d[19] + 2*d[2]*d[14]*d[19] + 2*d[5]*d[10]*d[20] - 2*d[4]*d[11]*d[20] - 2*d[2]*d[13]*d[20] + 2*d[1]*d[14]*d[20] + 2*d[7]*d[15]*d[21] + 2*d[6]*d[16]*d[21] - 2*d[2]*d[11]*d[22] + 2*d[5]*d[14]*d[22] - 2*d[6]*d[15]*d[22] + 2*d[7]*d[16]*d[22] + 2*d[2]*d[10]*d[23] + 2*d[1]*d[11]*d[23] + 2*d[5]*d[13]*d[23] + 2*d[4]*d[14]*d[23] + 2*d[7]*d[12]*d[24] - 2*d[6]*d[13]*d[24] - 2*d[4]*d[15]*d[24] + 2*d[3]*d[16]*d[24] + 2*d[6]*d[12]*d[25] + 2*d[7]*d[13]*d[25] + 2*d[3]*d[15]*d[25] + 2*d[4]*d[16]*d[25];
				coeffs[137] = 2*d[11]*d[14]*d[19] - 2*d[11]*d[13]*d[20] + 2*d[10]*d[14]*d[20] + 2*d[15]*d[16]*d[21] - std::pow(d[11],2)*d[22] + std::pow(d[14],2)*d[22] - std::pow(d[15],2)*d[22] + std::pow(d[16],2)*d[22] + 2*d[10]*d[11]*d[23] + 2*d[13]*d[14]*d[23] - 2*d[13]*d[15]*d[24] + 2*d[12]*d[16]*d[24] + 2*d[12]*d[15]*d[25] + 2*d[13]*d[16]*d[25];
				coeffs[138] = 2*d[8]*d[23]*d[25] - 2*d[8]*d[22]*d[26] + 2*d[7]*d[23]*d[26] + 2*d[5]*d[25]*d[26] - d[4]*std::pow(d[26],2);
				coeffs[139] = 2*d[17]*d[23]*d[25] - 2*d[17]*d[22]*d[26] + 2*d[16]*d[23]*d[26] + 2*d[14]*d[25]*d[26] - d[13]*std::pow(d[26],2);
				coeffs[140] = 2*d[1]*d[3]*d[18] - 2*d[0]*d[4]*d[18] + 2*d[0]*d[3]*d[19] + 2*d[1]*d[4]*d[19] + 2*d[0]*d[1]*d[21] + 2*d[3]*d[4]*d[21] - std::pow(d[0],2)*d[22] + std::pow(d[1],2)*d[22] + std::pow(d[3],2)*d[22] + 3*std::pow(d[4],2)*d[22];
				coeffs[141] = -2*d[4]*d[9]*d[18] + 2*d[3]*d[10]*d[18] + 2*d[1]*d[12]*d[18] - 2*d[0]*d[13]*d[18] + 2*d[3]*d[9]*d[19] + 2*d[4]*d[10]*d[19] + 2*d[0]*d[12]*d[19] + 2*d[1]*d[13]*d[19] + 2*d[1]*d[9]*d[21] + 2*d[0]*d[10]*d[21] + 2*d[4]*d[12]*d[21] + 2*d[3]*d[13]*d[21] - 2*d[0]*d[9]*d[22] + 2*d[1]*d[10]*d[22] + 2*d[3]*d[12]*d[22] + 6*d[4]*d[13]*d[22];
				coeffs[142] = 2*d[10]*d[12]*d[18] - 2*d[9]*d[13]*d[18] + 2*d[9]*d[12]*d[19] + 2*d[10]*d[13]*d[19] + 2*d[9]*d[10]*d[21] + 2*d[12]*d[13]*d[21] - std::pow(d[9],2)*d[22] + std::pow(d[10],2)*d[22] + std::pow(d[12],2)*d[22] + 3*std::pow(d[13],2)*d[22];
				coeffs[143] = 2*d[5]*d[19]*d[20] - d[4]*std::pow(d[20],2) - 2*d[2]*d[20]*d[22] + 2*d[2]*d[19]*d[23] + 2*d[1]*d[20]*d[23] + 2*d[5]*d[22]*d[23] + d[4]*std::pow(d[23],2) + 2*d[7]*d[21]*d[24] - 2*d[6]*d[22]*d[24] - d[4]*std::pow(d[24],2) + 2*d[6]*d[21]*d[25] + 2*d[7]*d[22]*d[25] + 2*d[3]*d[24]*d[25] + d[4]*std::pow(d[25],2);
				coeffs[144] = 2*d[14]*d[19]*d[20] - d[13]*std::pow(d[20],2) - 2*d[11]*d[20]*d[22] + 2*d[11]*d[19]*d[23] + 2*d[10]*d[20]*d[23] + 2*d[14]*d[22]*d[23] + d[13]*std::pow(d[23],2) + 2*d[16]*d[21]*d[24] - 2*d[15]*d[22]*d[24] - d[13]*std::pow(d[24],2) + 2*d[15]*d[21]*d[25] + 2*d[16]*d[22]*d[25] + 2*d[12]*d[24]*d[25] + d[13]*std::pow(d[25],2);
				coeffs[145] = 2*d[23]*d[25]*d[26] - d[22]*std::pow(d[26],2);
				coeffs[146] = -d[4]*std::pow(d[18],2) + 2*d[3]*d[18]*d[19] + d[4]*std::pow(d[19],2) + 2*d[1]*d[18]*d[21] + 2*d[0]*d[19]*d[21] + d[4]*std::pow(d[21],2) - 2*d[0]*d[18]*d[22] + 2*d[1]*d[19]*d[22] + 2*d[3]*d[21]*d[22] + 3*d[4]*std::pow(d[22],2);
				coeffs[147] = -d[13]*std::pow(d[18],2) + 2*d[12]*d[18]*d[19] + d[13]*std::pow(d[19],2) + 2*d[10]*d[18]*d[21] + 2*d[9]*d[19]*d[21] + d[13]*std::pow(d[21],2) - 2*d[9]*d[18]*d[22] + 2*d[10]*d[19]*d[22] + 2*d[12]*d[21]*d[22] + 3*d[13]*std::pow(d[22],2);
				coeffs[148] = -std::pow(d[20],2)*d[22] + 2*d[19]*d[20]*d[23] + d[22]*std::pow(d[23],2) - d[22]*std::pow(d[24],2) + 2*d[21]*d[24]*d[25] + d[22]*std::pow(d[25],2);
				coeffs[149] = 2*d[18]*d[19]*d[21] - std::pow(d[18],2)*d[22] + std::pow(d[19],2)*d[22] + std::pow(d[21],2)*d[22] + std::pow(d[22],3);
				coeffs[150] = d[5]*std::pow(d[8],2);
				coeffs[151] = std::pow(d[8],2)*d[14] + 2*d[5]*d[8]*d[17];
				coeffs[152] = 2*d[8]*d[14]*d[17] + d[5]*std::pow(d[17],2);
				coeffs[153] = d[14]*std::pow(d[17],2);
				coeffs[154] = std::pow(d[2],2)*d[5] + std::pow(d[5],3) - d[5]*std::pow(d[6],2) - d[5]*std::pow(d[7],2) + 2*d[3]*d[6]*d[8] + 2*d[4]*d[7]*d[8];
				coeffs[155] = 2*d[2]*d[5]*d[11] + 2*d[6]*d[8]*d[12] + 2*d[7]*d[8]*d[13] + std::pow(d[2],2)*d[14] + 3*std::pow(d[5],2)*d[14] - std::pow(d[6],2)*d[14] - std::pow(d[7],2)*d[14] - 2*d[5]*d[6]*d[15] + 2*d[3]*d[8]*d[15] - 2*d[5]*d[7]*d[16] + 2*d[4]*d[8]*d[16] + 2*d[3]*d[6]*d[17] + 2*d[4]*d[7]*d[17];
				coeffs[156] = d[5]*std::pow(d[11],2) + 2*d[2]*d[11]*d[14] + 3*d[5]*std::pow(d[14],2) + 2*d[8]*d[12]*d[15] - 2*d[6]*d[14]*d[15] - d[5]*std::pow(d[15],2) + 2*d[8]*d[13]*d[16] - 2*d[7]*d[14]*d[16] - d[5]*std::pow(d[16],2) + 2*d[6]*d[12]*d[17] + 2*d[7]*d[13]*d[17] + 2*d[3]*d[15]*d[17] + 2*d[4]*d[16]*d[17];
				coeffs[157] = std::pow(d[11],2)*d[14] + std::pow(d[14],3) - d[14]*std::pow(d[15],2) - d[14]*std::pow(d[16],2) + 2*d[12]*d[15]*d[17] + 2*d[13]*d[16]*d[17];
				coeffs[158] = std::pow(d[8],2)*d[23] + 2*d[5]*d[8]*d[26];
				coeffs[159] = 2*d[8]*d[17]*d[23] + 2*d[8]*d[14]*d[26] + 2*d[5]*d[17]*d[26];
				coeffs[160] = std::pow(d[17],2)*d[23] + 2*d[14]*d[17]*d[26];
				coeffs[161] = 2*d[0]*d[2]*d[3] + 2*d[1]*d[2]*d[4] - std::pow(d[0],2)*d[5] - std::pow(d[1],2)*d[5] + std::pow(d[3],2)*d[5] + std::pow(d[4],2)*d[5];
				coeffs[162] = 2*d[2]*d[3]*d[9] - 2*d[0]*d[5]*d[9] + 2*d[2]*d[4]*d[10] - 2*d[1]*d[5]*d[10] + 2*d[0]*d[3]*d[11] + 2*d[1]*d[4]*d[11] + 2*d[0]*d[2]*d[12] + 2*d[3]*d[5]*d[12] + 2*d[1]*d[2]*d[13] + 2*d[4]*d[5]*d[13] - std::pow(d[0],2)*d[14] - std::pow(d[1],2)*d[14] + std::pow(d[3],2)*d[14] + std::pow(d[4],2)*d[14];
				coeffs[163] = -d[5]*std::pow(d[9],2) - d[5]*std::pow(d[10],2) + 2*d[3]*d[9]*d[11] + 2*d[4]*d[10]*d[11] + 2*d[2]*d[9]*d[12] + 2*d[0]*d[11]*d[12] + d[5]*std::pow(d[12],2) + 2*d[2]*d[10]*d[13] + 2*d[1]*d[11]*d[13] + d[5]*std::pow(d[13],2) - 2*d[0]*d[9]*d[14] - 2*d[1]*d[10]*d[14] + 2*d[3]*d[12]*d[14] + 2*d[4]*d[13]*d[14];
				coeffs[164] = 2*d[9]*d[11]*d[12] + 2*d[10]*d[11]*d[13] - std::pow(d[9],2)*d[14] - std::pow(d[10],2)*d[14] + std::pow(d[12],2)*d[14] + std::pow(d[13],2)*d[14];
				coeffs[165] = 2*d[2]*d[5]*d[20] + 2*d[6]*d[8]*d[21] + 2*d[7]*d[8]*d[22] + std::pow(d[2],2)*d[23] + 3*std::pow(d[5],2)*d[23] - std::pow(d[6],2)*d[23] - std::pow(d[7],2)*d[23] - 2*d[5]*d[6]*d[24] + 2*d[3]*d[8]*d[24] - 2*d[5]*d[7]*d[25] + 2*d[4]*d[8]*d[25] + 2*d[3]*d[6]*d[26] + 2*d[4]*d[7]*d[26];
				coeffs[166] = 2*d[5]*d[11]*d[20] + 2*d[2]*d[14]*d[20] + 2*d[8]*d[15]*d[21] + 2*d[6]*d[17]*d[21] + 2*d[8]*d[16]*d[22] + 2*d[7]*d[17]*d[22] + 2*d[2]*d[11]*d[23] + 6*d[5]*d[14]*d[23] - 2*d[6]*d[15]*d[23] - 2*d[7]*d[16]*d[23] + 2*d[8]*d[12]*d[24] - 2*d[6]*d[14]*d[24] - 2*d[5]*d[15]*d[24] + 2*d[3]*d[17]*d[24] + 2*d[8]*d[13]*d[25] - 2*d[7]*d[14]*d[25] - 2*d[5]*d[16]*d[25] + 2*d[4]*d[17]*d[25] + 2*d[6]*d[12]*d[26] + 2*d[7]*d[13]*d[26] + 2*d[3]*d[15]*d[26] + 2*d[4]*d[16]*d[26];
				coeffs[167] = 2*d[11]*d[14]*d[20] + 2*d[15]*d[17]*d[21] + 2*d[16]*d[17]*d[22] + std::pow(d[11],2)*d[23] + 3*std::pow(d[14],2)*d[23] - std::pow(d[15],2)*d[23] - std::pow(d[16],2)*d[23] - 2*d[14]*d[15]*d[24] + 2*d[12]*d[17]*d[24] - 2*d[14]*d[16]*d[25] + 2*d[13]*d[17]*d[25] + 2*d[12]*d[15]*d[26] + 2*d[13]*d[16]*d[26];
				coeffs[168] = 2*d[8]*d[23]*d[26] + d[5]*std::pow(d[26],2);
				coeffs[169] = 2*d[17]*d[23]*d[26] + d[14]*std::pow(d[26],2);
				coeffs[170] = 2*d[2]*d[3]*d[18] - 2*d[0]*d[5]*d[18] + 2*d[2]*d[4]*d[19] - 2*d[1]*d[5]*d[19] + 2*d[0]*d[3]*d[20] + 2*d[1]*d[4]*d[20] + 2*d[0]*d[2]*d[21] + 2*d[3]*d[5]*d[21] + 2*d[1]*d[2]*d[22] + 2*d[4]*d[5]*d[22] - std::pow(d[0],2)*d[23] - std::pow(d[1],2)*d[23] + std::pow(d[3],2)*d[23] + std::pow(d[4],2)*d[23];
				coeffs[171] = -2*d[5]*d[9]*d[18] + 2*d[3]*d[11]*d[18] + 2*d[2]*d[12]*d[18] - 2*d[0]*d[14]*d[18] - 2*d[5]*d[10]*d[19] + 2*d[4]*d[11]*d[19] + 2*d[2]*d[13]*d[19] - 2*d[1]*d[14]*d[19] + 2*d[3]*d[9]*d[20] + 2*d[4]*d[10]*d[20] + 2*d[0]*d[12]*d[20] + 2*d[1]*d[13]*d[20] + 2*d[2]*d[9]*d[21] + 2*d[0]*d[11]*d[21] + 2*d[5]*d[12]*d[21] + 2*d[3]*d[14]*d[21] + 2*d[2]*d[10]*d[22] + 2*d[1]*d[11]*d[22] + 2*d[5]*d[13]*d[22] + 2*d[4]*d[14]*d[22] - 2*d[0]*d[9]*d[23] - 2*d[1]*d[10]*d[23] + 2*d[3]*d[12]*d[23] + 2*d[4]*d[13]*d[23];
				coeffs[172] = 2*d[11]*d[12]*d[18] - 2*d[9]*d[14]*d[18] + 2*d[11]*d[13]*d[19] - 2*d[10]*d[14]*d[19] + 2*d[9]*d[12]*d[20] + 2*d[10]*d[13]*d[20] + 2*d[9]*d[11]*d[21] + 2*d[12]*d[14]*d[21] + 2*d[10]*d[11]*d[22] + 2*d[13]*d[14]*d[22] - std::pow(d[9],2)*d[23] - std::pow(d[10],2)*d[23] + std::pow(d[12],2)*d[23] + std::pow(d[13],2)*d[23];
				coeffs[173] = d[5]*std::pow(d[20],2) + 2*d[2]*d[20]*d[23] + 3*d[5]*std::pow(d[23],2) + 2*d[8]*d[21]*d[24] - 2*d[6]*d[23]*d[24] - d[5]*std::pow(d[24],2) + 2*d[8]*d[22]*d[25] - 2*d[7]*d[23]*d[25] - d[5]*std::pow(d[25],2) + 2*d[6]*d[21]*d[26] + 2*d[7]*d[22]*d[26] + 2*d[3]*d[24]*d[26] + 2*d[4]*d[25]*d[26];
				coeffs[174] = d[14]*std::pow(d[20],2) + 2*d[11]*d[20]*d[23] + 3*d[14]*std::pow(d[23],2) + 2*d[17]*d[21]*d[24] - 2*d[15]*d[23]*d[24] - d[14]*std::pow(d[24],2) + 2*d[17]*d[22]*d[25] - 2*d[16]*d[23]*d[25] - d[14]*std::pow(d[25],2) + 2*d[15]*d[21]*d[26] + 2*d[16]*d[22]*d[26] + 2*d[12]*d[24]*d[26] + 2*d[13]*d[25]*d[26];
				coeffs[175] = d[23]*std::pow(d[26],2);
				coeffs[176] = -d[5]*std::pow(d[18],2) - d[5]*std::pow(d[19],2) + 2*d[3]*d[18]*d[20] + 2*d[4]*d[19]*d[20] + 2*d[2]*d[18]*d[21] + 2*d[0]*d[20]*d[21] + d[5]*std::pow(d[21],2) + 2*d[2]*d[19]*d[22] + 2*d[1]*d[20]*d[22] + d[5]*std::pow(d[22],2) - 2*d[0]*d[18]*d[23] - 2*d[1]*d[19]*d[23] + 2*d[3]*d[21]*d[23] + 2*d[4]*d[22]*d[23];
				coeffs[177] = -d[14]*std::pow(d[18],2) - d[14]*std::pow(d[19],2) + 2*d[12]*d[18]*d[20] + 2*d[13]*d[19]*d[20] + 2*d[11]*d[18]*d[21] + 2*d[9]*d[20]*d[21] + d[14]*std::pow(d[21],2) + 2*d[11]*d[19]*d[22] + 2*d[10]*d[20]*d[22] + d[14]*std::pow(d[22],2) - 2*d[9]*d[18]*d[23] - 2*d[10]*d[19]*d[23] + 2*d[12]*d[21]*d[23] + 2*d[13]*d[22]*d[23];
				coeffs[178] = std::pow(d[20],2)*d[23] + std::pow(d[23],3) - d[23]*std::pow(d[24],2) - d[23]*std::pow(d[25],2) + 2*d[21]*d[24]*d[26] + 2*d[22]*d[25]*d[26];
				coeffs[179] = 2*d[18]*d[20]*d[21] + 2*d[19]*d[20]*d[22] - std::pow(d[18],2)*d[23] - std::pow(d[19],2)*d[23] + std::pow(d[21],2)*d[23] + std::pow(d[22],2)*d[23];
				coeffs[180] = d[6]*std::pow(d[8],2);
				coeffs[181] = std::pow(d[8],2)*d[15] + 2*d[6]*d[8]*d[17];
				coeffs[182] = 2*d[8]*d[15]*d[17] + d[6]*std::pow(d[17],2);
				coeffs[183] = d[15]*std::pow(d[17],2);
				coeffs[184] = -std::pow(d[2],2)*d[6] - std::pow(d[5],2)*d[6] + std::pow(d[6],3) + d[6]*std::pow(d[7],2) + 2*d[0]*d[2]*d[8] + 2*d[3]*d[5]*d[8];
				coeffs[185] = 2*d[2]*d[8]*d[9] - 2*d[2]*d[6]*d[11] + 2*d[0]*d[8]*d[11] + 2*d[5]*d[8]*d[12] - 2*d[5]*d[6]*d[14] + 2*d[3]*d[8]*d[14] - std::pow(d[2],2)*d[15] - std::pow(d[5],2)*d[15] + 3*std::pow(d[6],2)*d[15] + std::pow(d[7],2)*d[15] + 2*d[6]*d[7]*d[16] + 2*d[0]*d[2]*d[17] + 2*d[3]*d[5]*d[17];
				coeffs[186] = 2*d[8]*d[9]*d[11] - d[6]*std::pow(d[11],2) + 2*d[8]*d[12]*d[14] - d[6]*std::pow(d[14],2) - 2*d[2]*d[11]*d[15] - 2*d[5]*d[14]*d[15] + 3*d[6]*std::pow(d[15],2) + 2*d[7]*d[15]*d[16] + d[6]*std::pow(d[16],2) + 2*d[2]*d[9]*d[17] + 2*d[0]*d[11]*d[17] + 2*d[5]*d[12]*d[17] + 2*d[3]*d[14]*d[17];
				coeffs[187] = -std::pow(d[11],2)*d[15] - std::pow(d[14],2)*d[15] + std::pow(d[15],3) + d[15]*std::pow(d[16],2) + 2*d[9]*d[11]*d[17] + 2*d[12]*d[14]*d[17];
				coeffs[188] = std::pow(d[8],2)*d[24] + 2*d[6]*d[8]*d[26];
				coeffs[189] = 2*d[8]*d[17]*d[24] + 2*d[8]*d[15]*d[26] + 2*d[6]*d[17]*d[26];
				coeffs[190] = std::pow(d[17],2)*d[24] + 2*d[15]*d[17]*d[26];
				coeffs[191] = std::pow(d[0],2)*d[6] - std::pow(d[1],2)*d[6] + std::pow(d[3],2)*d[6] - std::pow(d[4],2)*d[6] + 2*d[0]*d[1]*d[7] + 2*d[3]*d[4]*d[7];
				coeffs[192] = 2*d[0]*d[6]*d[9] + 2*d[1]*d[7]*d[9] - 2*d[1]*d[6]*d[10] + 2*d[0]*d[7]*d[10] + 2*d[3]*d[6]*d[12] + 2*d[4]*d[7]*d[12] - 2*d[4]*d[6]*d[13] + 2*d[3]*d[7]*d[13] + std::pow(d[0],2)*d[15] - std::pow(d[1],2)*d[15] + std::pow(d[3],2)*d[15] - std::pow(d[4],2)*d[15] + 2*d[0]*d[1]*d[16] + 2*d[3]*d[4]*d[16];
				coeffs[193] = d[6]*std::pow(d[9],2) + 2*d[7]*d[9]*d[10] - d[6]*std::pow(d[10],2) + d[6]*std::pow(d[12],2) + 2*d[7]*d[12]*d[13] - d[6]*std::pow(d[13],2) + 2*d[0]*d[9]*d[15] - 2*d[1]*d[10]*d[15] + 2*d[3]*d[12]*d[15] - 2*d[4]*d[13]*d[15] + 2*d[1]*d[9]*d[16] + 2*d[0]*d[10]*d[16] + 2*d[4]*d[12]*d[16] + 2*d[3]*d[13]*d[16];
				coeffs[194] = std::pow(d[9],2)*d[15] - std::pow(d[10],2)*d[15] + std::pow(d[12],2)*d[15] - std::pow(d[13],2)*d[15] + 2*d[9]*d[10]*d[16] + 2*d[12]*d[13]*d[16];
				coeffs[195] = 2*d[2]*d[8]*d[18] - 2*d[2]*d[6]*d[20] + 2*d[0]*d[8]*d[20] + 2*d[5]*d[8]*d[21] - 2*d[5]*d[6]*d[23] + 2*d[3]*d[8]*d[23] - std::pow(d[2],2)*d[24] - std::pow(d[5],2)*d[24] + 3*std::pow(d[6],2)*d[24] + std::pow(d[7],2)*d[24] + 2*d[6]*d[7]*d[25] + 2*d[0]*d[2]*d[26] + 2*d[3]*d[5]*d[26];
				coeffs[196] = 2*d[8]*d[11]*d[18] + 2*d[2]*d[17]*d[18] + 2*d[8]*d[9]*d[20] - 2*d[6]*d[11]*d[20] - 2*d[2]*d[15]*d[20] + 2*d[0]*d[17]*d[20] + 2*d[8]*d[14]*d[21] + 2*d[5]*d[17]*d[21] + 2*d[8]*d[12]*d[23] - 2*d[6]*d[14]*d[23] - 2*d[5]*d[15]*d[23] + 2*d[3]*d[17]*d[23] - 2*d[2]*d[11]*d[24] - 2*d[5]*d[14]*d[24] + 6*d[6]*d[15]*d[24] + 2*d[7]*d[16]*d[24] + 2*d[7]*d[15]*d[25] + 2*d[6]*d[16]*d[25] + 2*d[2]*d[9]*d[26] + 2*d[0]*d[11]*d[26] + 2*d[5]*d[12]*d[26] + 2*d[3]*d[14]*d[26];
				coeffs[197] = 2*d[11]*d[17]*d[18] - 2*d[11]*d[15]*d[20] + 2*d[9]*d[17]*d[20] + 2*d[14]*d[17]*d[21] - 2*d[14]*d[15]*d[23] + 2*d[12]*d[17]*d[23] - std::pow(d[11],2)*d[24] - std::pow(d[14],2)*d[24] + 3*std::pow(d[15],2)*d[24] + std::pow(d[16],2)*d[24] + 2*d[15]*d[16]*d[25] + 2*d[9]*d[11]*d[26] + 2*d[12]*d[14]*d[26];
				coeffs[198] = 2*d[8]*d[24]*d[26] + d[6]*std::pow(d[26],2);
				coeffs[199] = 2*d[17]*d[24]*d[26] + d[15]*std::pow(d[26],2);
				coeffs[200] = 2*d[0]*d[6]*d[18] + 2*d[1]*d[7]*d[18] - 2*d[1]*d[6]*d[19] + 2*d[0]*d[7]*d[19] + 2*d[3]*d[6]*d[21] + 2*d[4]*d[7]*d[21] - 2*d[4]*d[6]*d[22] + 2*d[3]*d[7]*d[22] + std::pow(d[0],2)*d[24] - std::pow(d[1],2)*d[24] + std::pow(d[3],2)*d[24] - std::pow(d[4],2)*d[24] + 2*d[0]*d[1]*d[25] + 2*d[3]*d[4]*d[25];
				coeffs[201] = 2*d[6]*d[9]*d[18] + 2*d[7]*d[10]*d[18] + 2*d[0]*d[15]*d[18] + 2*d[1]*d[16]*d[18] + 2*d[7]*d[9]*d[19] - 2*d[6]*d[10]*d[19] - 2*d[1]*d[15]*d[19] + 2*d[0]*d[16]*d[19] + 2*d[6]*d[12]*d[21] + 2*d[7]*d[13]*d[21] + 2*d[3]*d[15]*d[21] + 2*d[4]*d[16]*d[21] + 2*d[7]*d[12]*d[22] - 2*d[6]*d[13]*d[22] - 2*d[4]*d[15]*d[22] + 2*d[3]*d[16]*d[22] + 2*d[0]*d[9]*d[24] - 2*d[1]*d[10]*d[24] + 2*d[3]*d[12]*d[24] - 2*d[4]*d[13]*d[24] + 2*d[1]*d[9]*d[25] + 2*d[0]*d[10]*d[25] + 2*d[4]*d[12]*d[25] + 2*d[3]*d[13]*d[25];
				coeffs[202] = 2*d[9]*d[15]*d[18] + 2*d[10]*d[16]*d[18] - 2*d[10]*d[15]*d[19] + 2*d[9]*d[16]*d[19] + 2*d[12]*d[15]*d[21] + 2*d[13]*d[16]*d[21] - 2*d[13]*d[15]*d[22] + 2*d[12]*d[16]*d[22] + std::pow(d[9],2)*d[24] - std::pow(d[10],2)*d[24] + std::pow(d[12],2)*d[24] - std::pow(d[13],2)*d[24] + 2*d[9]*d[10]*d[25] + 2*d[12]*d[13]*d[25];
				coeffs[203] = 2*d[8]*d[18]*d[20] - d[6]*std::pow(d[20],2) + 2*d[8]*d[21]*d[23] - d[6]*std::pow(d[23],2) - 2*d[2]*d[20]*d[24] - 2*d[5]*d[23]*d[24] + 3*d[6]*std::pow(d[24],2) + 2*d[7]*d[24]*d[25] + d[6]*std::pow(d[25],2) + 2*d[2]*d[18]*d[26] + 2*d[0]*d[20]*d[26] + 2*d[5]*d[21]*d[26] + 2*d[3]*d[23]*d[26];
				coeffs[204] = 2*d[17]*d[18]*d[20] - d[15]*std::pow(d[20],2) + 2*d[17]*d[21]*d[23] - d[15]*std::pow(d[23],2) - 2*d[11]*d[20]*d[24] - 2*d[14]*d[23]*d[24] + 3*d[15]*std::pow(d[24],2) + 2*d[16]*d[24]*d[25] + d[15]*std::pow(d[25],2) + 2*d[11]*d[18]*d[26] + 2*d[9]*d[20]*d[26] + 2*d[14]*d[21]*d[26] + 2*d[12]*d[23]*d[26];
				coeffs[205] = d[24]*std::pow(d[26],2);
				coeffs[206] = d[6]*std::pow(d[18],2) + 2*d[7]*d[18]*d[19] - d[6]*std::pow(d[19],2) + d[6]*std::pow(d[21],2) + 2*d[7]*d[21]*d[22] - d[6]*std::pow(d[22],2) + 2*d[0]*d[18]*d[24] - 2*d[1]*d[19]*d[24] + 2*d[3]*d[21]*d[24] - 2*d[4]*d[22]*d[24] + 2*d[1]*d[18]*d[25] + 2*d[0]*d[19]*d[25] + 2*d[4]*d[21]*d[25] + 2*d[3]*d[22]*d[25];
				coeffs[207] = d[15]*std::pow(d[18],2) + 2*d[16]*d[18]*d[19] - d[15]*std::pow(d[19],2) + d[15]*std::pow(d[21],2) + 2*d[16]*d[21]*d[22] - d[15]*std::pow(d[22],2) + 2*d[9]*d[18]*d[24] - 2*d[10]*d[19]*d[24] + 2*d[12]*d[21]*d[24] - 2*d[13]*d[22]*d[24] + 2*d[10]*d[18]*d[25] + 2*d[9]*d[19]*d[25] + 2*d[13]*d[21]*d[25] + 2*d[12]*d[22]*d[25];
				coeffs[208] = -std::pow(d[20],2)*d[24] - std::pow(d[23],2)*d[24] + std::pow(d[24],3) + d[24]*std::pow(d[25],2) + 2*d[18]*d[20]*d[26] + 2*d[21]*d[23]*d[26];
				coeffs[209] = std::pow(d[18],2)*d[24] - std::pow(d[19],2)*d[24] + std::pow(d[21],2)*d[24] - std::pow(d[22],2)*d[24] + 2*d[18]*d[19]*d[25] + 2*d[21]*d[22]*d[25];
				coeffs[210] = d[7]*std::pow(d[8],2);
				coeffs[211] = std::pow(d[8],2)*d[16] + 2*d[7]*d[8]*d[17];
				coeffs[212] = 2*d[8]*d[16]*d[17] + d[7]*std::pow(d[17],2);
				coeffs[213] = d[16]*std::pow(d[17],2);
				coeffs[214] = -std::pow(d[2],2)*d[7] - std::pow(d[5],2)*d[7] + std::pow(d[6],2)*d[7] + std::pow(d[7],3) + 2*d[1]*d[2]*d[8] + 2*d[4]*d[5]*d[8];
				coeffs[215] = 2*d[2]*d[8]*d[10] - 2*d[2]*d[7]*d[11] + 2*d[1]*d[8]*d[11] + 2*d[5]*d[8]*d[13] - 2*d[5]*d[7]*d[14] + 2*d[4]*d[8]*d[14] + 2*d[6]*d[7]*d[15] - std::pow(d[2],2)*d[16] - std::pow(d[5],2)*d[16] + std::pow(d[6],2)*d[16] + 3*std::pow(d[7],2)*d[16] + 2*d[1]*d[2]*d[17] + 2*d[4]*d[5]*d[17];
				coeffs[216] = 2*d[8]*d[10]*d[11] - d[7]*std::pow(d[11],2) + 2*d[8]*d[13]*d[14] - d[7]*std::pow(d[14],2) + d[7]*std::pow(d[15],2) - 2*d[2]*d[11]*d[16] - 2*d[5]*d[14]*d[16] + 2*d[6]*d[15]*d[16] + 3*d[7]*std::pow(d[16],2) + 2*d[2]*d[10]*d[17] + 2*d[1]*d[11]*d[17] + 2*d[5]*d[13]*d[17] + 2*d[4]*d[14]*d[17];
				coeffs[217] = -std::pow(d[11],2)*d[16] - std::pow(d[14],2)*d[16] + std::pow(d[15],2)*d[16] + std::pow(d[16],3) + 2*d[10]*d[11]*d[17] + 2*d[13]*d[14]*d[17];
				coeffs[218] = std::pow(d[8],2)*d[25] + 2*d[7]*d[8]*d[26];
				coeffs[219] = 2*d[8]*d[17]*d[25] + 2*d[8]*d[16]*d[26] + 2*d[7]*d[17]*d[26];
				coeffs[220] = std::pow(d[17],2)*d[25] + 2*d[16]*d[17]*d[26];
				coeffs[221] = 2*d[0]*d[1]*d[6] + 2*d[3]*d[4]*d[6] - std::pow(d[0],2)*d[7] + std::pow(d[1],2)*d[7] - std::pow(d[3],2)*d[7] + std::pow(d[4],2)*d[7];
				coeffs[222] = 2*d[1]*d[6]*d[9] - 2*d[0]*d[7]*d[9] + 2*d[0]*d[6]*d[10] + 2*d[1]*d[7]*d[10] + 2*d[4]*d[6]*d[12] - 2*d[3]*d[7]*d[12] + 2*d[3]*d[6]*d[13] + 2*d[4]*d[7]*d[13] + 2*d[0]*d[1]*d[15] + 2*d[3]*d[4]*d[15] - std::pow(d[0],2)*d[16] + std::pow(d[1],2)*d[16] - std::pow(d[3],2)*d[16] + std::pow(d[4],2)*d[16];
				coeffs[223] = -d[7]*std::pow(d[9],2) + 2*d[6]*d[9]*d[10] + d[7]*std::pow(d[10],2) - d[7]*std::pow(d[12],2) + 2*d[6]*d[12]*d[13] + d[7]*std::pow(d[13],2) + 2*d[1]*d[9]*d[15] + 2*d[0]*d[10]*d[15] + 2*d[4]*d[12]*d[15] + 2*d[3]*d[13]*d[15] - 2*d[0]*d[9]*d[16] + 2*d[1]*d[10]*d[16] - 2*d[3]*d[12]*d[16] + 2*d[4]*d[13]*d[16];
				coeffs[224] = 2*d[9]*d[10]*d[15] + 2*d[12]*d[13]*d[15] - std::pow(d[9],2)*d[16] + std::pow(d[10],2)*d[16] - std::pow(d[12],2)*d[16] + std::pow(d[13],2)*d[16];
				coeffs[225] = 2*d[2]*d[8]*d[19] - 2*d[2]*d[7]*d[20] + 2*d[1]*d[8]*d[20] + 2*d[5]*d[8]*d[22] - 2*d[5]*d[7]*d[23] + 2*d[4]*d[8]*d[23] + 2*d[6]*d[7]*d[24] - std::pow(d[2],2)*d[25] - std::pow(d[5],2)*d[25] + std::pow(d[6],2)*d[25] + 3*std::pow(d[7],2)*d[25] + 2*d[1]*d[2]*d[26] + 2*d[4]*d[5]*d[26];
				coeffs[226] = 2*d[8]*d[11]*d[19] + 2*d[2]*d[17]*d[19] + 2*d[8]*d[10]*d[20] - 2*d[7]*d[11]*d[20] - 2*d[2]*d[16]*d[20] + 2*d[1]*d[17]*d[20] + 2*d[8]*d[14]*d[22] + 2*d[5]*d[17]*d[22] + 2*d[8]*d[13]*d[23] - 2*d[7]*d[14]*d[23] - 2*d[5]*d[16]*d[23] + 2*d[4]*d[17]*d[23] + 2*d[7]*d[15]*d[24] + 2*d[6]*d[16]*d[24] - 2*d[2]*d[11]*d[25] - 2*d[5]*d[14]*d[25] + 2*d[6]*d[15]*d[25] + 6*d[7]*d[16]*d[25] + 2*d[2]*d[10]*d[26] + 2*d[1]*d[11]*d[26] + 2*d[5]*d[13]*d[26] + 2*d[4]*d[14]*d[26];
				coeffs[227] = 2*d[11]*d[17]*d[19] - 2*d[11]*d[16]*d[20] + 2*d[10]*d[17]*d[20] + 2*d[14]*d[17]*d[22] - 2*d[14]*d[16]*d[23] + 2*d[13]*d[17]*d[23] + 2*d[15]*d[16]*d[24] - std::pow(d[11],2)*d[25] - std::pow(d[14],2)*d[25] + std::pow(d[15],2)*d[25] + 3*std::pow(d[16],2)*d[25] + 2*d[10]*d[11]*d[26] + 2*d[13]*d[14]*d[26];
				coeffs[228] = 2*d[8]*d[25]*d[26] + d[7]*std::pow(d[26],2);
				coeffs[229] = 2*d[17]*d[25]*d[26] + d[16]*std::pow(d[26],2);
				coeffs[230] = 2*d[1]*d[6]*d[18] - 2*d[0]*d[7]*d[18] + 2*d[0]*d[6]*d[19] + 2*d[1]*d[7]*d[19] + 2*d[4]*d[6]*d[21] - 2*d[3]*d[7]*d[21] + 2*d[3]*d[6]*d[22] + 2*d[4]*d[7]*d[22] + 2*d[0]*d[1]*d[24] + 2*d[3]*d[4]*d[24] - std::pow(d[0],2)*d[25] + std::pow(d[1],2)*d[25] - std::pow(d[3],2)*d[25] + std::pow(d[4],2)*d[25];
				coeffs[231] = -2*d[7]*d[9]*d[18] + 2*d[6]*d[10]*d[18] + 2*d[1]*d[15]*d[18] - 2*d[0]*d[16]*d[18] + 2*d[6]*d[9]*d[19] + 2*d[7]*d[10]*d[19] + 2*d[0]*d[15]*d[19] + 2*d[1]*d[16]*d[19] - 2*d[7]*d[12]*d[21] + 2*d[6]*d[13]*d[21] + 2*d[4]*d[15]*d[21] - 2*d[3]*d[16]*d[21] + 2*d[6]*d[12]*d[22] + 2*d[7]*d[13]*d[22] + 2*d[3]*d[15]*d[22] + 2*d[4]*d[16]*d[22] + 2*d[1]*d[9]*d[24] + 2*d[0]*d[10]*d[24] + 2*d[4]*d[12]*d[24] + 2*d[3]*d[13]*d[24] - 2*d[0]*d[9]*d[25] + 2*d[1]*d[10]*d[25] - 2*d[3]*d[12]*d[25] + 2*d[4]*d[13]*d[25];
				coeffs[232] = 2*d[10]*d[15]*d[18] - 2*d[9]*d[16]*d[18] + 2*d[9]*d[15]*d[19] + 2*d[10]*d[16]*d[19] + 2*d[13]*d[15]*d[21] - 2*d[12]*d[16]*d[21] + 2*d[12]*d[15]*d[22] + 2*d[13]*d[16]*d[22] + 2*d[9]*d[10]*d[24] + 2*d[12]*d[13]*d[24] - std::pow(d[9],2)*d[25] + std::pow(d[10],2)*d[25] - std::pow(d[12],2)*d[25] + std::pow(d[13],2)*d[25];
				coeffs[233] = 2*d[8]*d[19]*d[20] - d[7]*std::pow(d[20],2) + 2*d[8]*d[22]*d[23] - d[7]*std::pow(d[23],2) + d[7]*std::pow(d[24],2) - 2*d[2]*d[20]*d[25] - 2*d[5]*d[23]*d[25] + 2*d[6]*d[24]*d[25] + 3*d[7]*std::pow(d[25],2) + 2*d[2]*d[19]*d[26] + 2*d[1]*d[20]*d[26] + 2*d[5]*d[22]*d[26] + 2*d[4]*d[23]*d[26];
				coeffs[234] = 2*d[17]*d[19]*d[20] - d[16]*std::pow(d[20],2) + 2*d[17]*d[22]*d[23] - d[16]*std::pow(d[23],2) + d[16]*std::pow(d[24],2) - 2*d[11]*d[20]*d[25] - 2*d[14]*d[23]*d[25] + 2*d[15]*d[24]*d[25] + 3*d[16]*std::pow(d[25],2) + 2*d[11]*d[19]*d[26] + 2*d[10]*d[20]*d[26] + 2*d[14]*d[22]*d[26] + 2*d[13]*d[23]*d[26];
				coeffs[235] = d[25]*std::pow(d[26],2);
				coeffs[236] = -d[7]*std::pow(d[18],2) + 2*d[6]*d[18]*d[19] + d[7]*std::pow(d[19],2) - d[7]*std::pow(d[21],2) + 2*d[6]*d[21]*d[22] + d[7]*std::pow(d[22],2) + 2*d[1]*d[18]*d[24] + 2*d[0]*d[19]*d[24] + 2*d[4]*d[21]*d[24] + 2*d[3]*d[22]*d[24] - 2*d[0]*d[18]*d[25] + 2*d[1]*d[19]*d[25] - 2*d[3]*d[21]*d[25] + 2*d[4]*d[22]*d[25];
				coeffs[237] = -d[16]*std::pow(d[18],2) + 2*d[15]*d[18]*d[19] + d[16]*std::pow(d[19],2) - d[16]*std::pow(d[21],2) + 2*d[15]*d[21]*d[22] + d[16]*std::pow(d[22],2) + 2*d[10]*d[18]*d[24] + 2*d[9]*d[19]*d[24] + 2*d[13]*d[21]*d[24] + 2*d[12]*d[22]*d[24] - 2*d[9]*d[18]*d[25] + 2*d[10]*d[19]*d[25] - 2*d[12]*d[21]*d[25] + 2*d[13]*d[22]*d[25];
				coeffs[238] = -std::pow(d[20],2)*d[25] - std::pow(d[23],2)*d[25] + std::pow(d[24],2)*d[25] + std::pow(d[25],3) + 2*d[19]*d[20]*d[26] + 2*d[22]*d[23]*d[26];
				coeffs[239] = 2*d[18]*d[19]*d[24] + 2*d[21]*d[22]*d[24] - std::pow(d[18],2)*d[25] + std::pow(d[19],2)*d[25] - std::pow(d[21],2)*d[25] + std::pow(d[22],2)*d[25];
				coeffs[240] = std::pow(d[8],3);
				coeffs[241] = 3*std::pow(d[8],2)*d[17];
				coeffs[242] = 3*d[8]*std::pow(d[17],2);
				coeffs[243] = std::pow(d[17],3);
				coeffs[244] = std::pow(d[2],2)*d[8] + std::pow(d[5],2)*d[8] + std::pow(d[6],2)*d[8] + std::pow(d[7],2)*d[8];
				coeffs[245] = 2*d[2]*d[8]*d[11] + 2*d[5]*d[8]*d[14] + 2*d[6]*d[8]*d[15] + 2*d[7]*d[8]*d[16] + std::pow(d[2],2)*d[17] + std::pow(d[5],2)*d[17] + std::pow(d[6],2)*d[17] + std::pow(d[7],2)*d[17];
				coeffs[246] = d[8]*std::pow(d[11],2) + d[8]*std::pow(d[14],2) + d[8]*std::pow(d[15],2) + d[8]*std::pow(d[16],2) + 2*d[2]*d[11]*d[17] + 2*d[5]*d[14]*d[17] + 2*d[6]*d[15]*d[17] + 2*d[7]*d[16]*d[17];
				coeffs[247] = std::pow(d[11],2)*d[17] + std::pow(d[14],2)*d[17] + std::pow(d[15],2)*d[17] + std::pow(d[16],2)*d[17];
				coeffs[248] = 3*std::pow(d[8],2)*d[26];
				coeffs[249] = 6*d[8]*d[17]*d[26];
				coeffs[250] = 3*std::pow(d[17],2)*d[26];
				coeffs[251] = 2*d[0]*d[2]*d[6] + 2*d[3]*d[5]*d[6] + 2*d[1]*d[2]*d[7] + 2*d[4]*d[5]*d[7] - std::pow(d[0],2)*d[8] - std::pow(d[1],2)*d[8] - std::pow(d[3],2)*d[8] - std::pow(d[4],2)*d[8];
				coeffs[252] = 2*d[2]*d[6]*d[9] - 2*d[0]*d[8]*d[9] + 2*d[2]*d[7]*d[10] - 2*d[1]*d[8]*d[10] + 2*d[0]*d[6]*d[11] + 2*d[1]*d[7]*d[11] + 2*d[5]*d[6]*d[12] - 2*d[3]*d[8]*d[12] + 2*d[5]*d[7]*d[13] - 2*d[4]*d[8]*d[13] + 2*d[3]*d[6]*d[14] + 2*d[4]*d[7]*d[14] + 2*d[0]*d[2]*d[15] + 2*d[3]*d[5]*d[15] + 2*d[1]*d[2]*d[16] + 2*d[4]*d[5]*d[16] - std::pow(d[0],2)*d[17] - std::pow(d[1],2)*d[17] - std::pow(d[3],2)*d[17] - std::pow(d[4],2)*d[17];
				coeffs[253] = -d[8]*std::pow(d[9],2) - d[8]*std::pow(d[10],2) + 2*d[6]*d[9]*d[11] + 2*d[7]*d[10]*d[11] - d[8]*std::pow(d[12],2) - d[8]*std::pow(d[13],2) + 2*d[6]*d[12]*d[14] + 2*d[7]*d[13]*d[14] + 2*d[2]*d[9]*d[15] + 2*d[0]*d[11]*d[15] + 2*d[5]*d[12]*d[15] + 2*d[3]*d[14]*d[15] + 2*d[2]*d[10]*d[16] + 2*d[1]*d[11]*d[16] + 2*d[5]*d[13]*d[16] + 2*d[4]*d[14]*d[16] - 2*d[0]*d[9]*d[17] - 2*d[1]*d[10]*d[17] - 2*d[3]*d[12]*d[17] - 2*d[4]*d[13]*d[17];
				coeffs[254] = 2*d[9]*d[11]*d[15] + 2*d[12]*d[14]*d[15] + 2*d[10]*d[11]*d[16] + 2*d[13]*d[14]*d[16] - std::pow(d[9],2)*d[17] - std::pow(d[10],2)*d[17] - std::pow(d[12],2)*d[17] - std::pow(d[13],2)*d[17];
				coeffs[255] = 2*d[2]*d[8]*d[20] + 2*d[5]*d[8]*d[23] + 2*d[6]*d[8]*d[24] + 2*d[7]*d[8]*d[25] + std::pow(d[2],2)*d[26] + std::pow(d[5],2)*d[26] + std::pow(d[6],2)*d[26] + std::pow(d[7],2)*d[26];
				coeffs[256] = 2*d[8]*d[11]*d[20] + 2*d[2]*d[17]*d[20] + 2*d[8]*d[14]*d[23] + 2*d[5]*d[17]*d[23] + 2*d[8]*d[15]*d[24] + 2*d[6]*d[17]*d[24] + 2*d[8]*d[16]*d[25] + 2*d[7]*d[17]*d[25] + 2*d[2]*d[11]*d[26] + 2*d[5]*d[14]*d[26] + 2*d[6]*d[15]*d[26] + 2*d[7]*d[16]*d[26];
				coeffs[257] = 2*d[11]*d[17]*d[20] + 2*d[14]*d[17]*d[23] + 2*d[15]*d[17]*d[24] + 2*d[16]*d[17]*d[25] + std::pow(d[11],2)*d[26] + std::pow(d[14],2)*d[26] + std::pow(d[15],2)*d[26] + std::pow(d[16],2)*d[26];
				coeffs[258] = 3*d[8]*std::pow(d[26],2);
				coeffs[259] = 3*d[17]*std::pow(d[26],2);
				coeffs[260] = 2*d[2]*d[6]*d[18] - 2*d[0]*d[8]*d[18] + 2*d[2]*d[7]*d[19] - 2*d[1]*d[8]*d[19] + 2*d[0]*d[6]*d[20] + 2*d[1]*d[7]*d[20] + 2*d[5]*d[6]*d[21] - 2*d[3]*d[8]*d[21] + 2*d[5]*d[7]*d[22] - 2*d[4]*d[8]*d[22] + 2*d[3]*d[6]*d[23] + 2*d[4]*d[7]*d[23] + 2*d[0]*d[2]*d[24] + 2*d[3]*d[5]*d[24] + 2*d[1]*d[2]*d[25] + 2*d[4]*d[5]*d[25] - std::pow(d[0],2)*d[26] - std::pow(d[1],2)*d[26] - std::pow(d[3],2)*d[26] - std::pow(d[4],2)*d[26];
				coeffs[261] = -2*d[8]*d[9]*d[18] + 2*d[6]*d[11]*d[18] + 2*d[2]*d[15]*d[18] - 2*d[0]*d[17]*d[18] - 2*d[8]*d[10]*d[19] + 2*d[7]*d[11]*d[19] + 2*d[2]*d[16]*d[19] - 2*d[1]*d[17]*d[19] + 2*d[6]*d[9]*d[20] + 2*d[7]*d[10]*d[20] + 2*d[0]*d[15]*d[20] + 2*d[1]*d[16]*d[20] - 2*d[8]*d[12]*d[21] + 2*d[6]*d[14]*d[21] + 2*d[5]*d[15]*d[21] - 2*d[3]*d[17]*d[21] - 2*d[8]*d[13]*d[22] + 2*d[7]*d[14]*d[22] + 2*d[5]*d[16]*d[22] - 2*d[4]*d[17]*d[22] + 2*d[6]*d[12]*d[23] + 2*d[7]*d[13]*d[23] + 2*d[3]*d[15]*d[23] + 2*d[4]*d[16]*d[23] + 2*d[2]*d[9]*d[24] + 2*d[0]*d[11]*d[24] + 2*d[5]*d[12]*d[24] + 2*d[3]*d[14]*d[24] + 2*d[2]*d[10]*d[25] + 2*d[1]*d[11]*d[25] + 2*d[5]*d[13]*d[25] + 2*d[4]*d[14]*d[25] - 2*d[0]*d[9]*d[26] - 2*d[1]*d[10]*d[26] - 2*d[3]*d[12]*d[26] - 2*d[4]*d[13]*d[26];
				coeffs[262] = 2*d[11]*d[15]*d[18] - 2*d[9]*d[17]*d[18] + 2*d[11]*d[16]*d[19] - 2*d[10]*d[17]*d[19] + 2*d[9]*d[15]*d[20] + 2*d[10]*d[16]*d[20] + 2*d[14]*d[15]*d[21] - 2*d[12]*d[17]*d[21] + 2*d[14]*d[16]*d[22] - 2*d[13]*d[17]*d[22] + 2*d[12]*d[15]*d[23] + 2*d[13]*d[16]*d[23] + 2*d[9]*d[11]*d[24] + 2*d[12]*d[14]*d[24] + 2*d[10]*d[11]*d[25] + 2*d[13]*d[14]*d[25] - std::pow(d[9],2)*d[26] - std::pow(d[10],2)*d[26] - std::pow(d[12],2)*d[26] - std::pow(d[13],2)*d[26];
				coeffs[263] = d[8]*std::pow(d[20],2) + d[8]*std::pow(d[23],2) + d[8]*std::pow(d[24],2) + d[8]*std::pow(d[25],2) + 2*d[2]*d[20]*d[26] + 2*d[5]*d[23]*d[26] + 2*d[6]*d[24]*d[26] + 2*d[7]*d[25]*d[26];
				coeffs[264] = d[17]*std::pow(d[20],2) + d[17]*std::pow(d[23],2) + d[17]*std::pow(d[24],2) + d[17]*std::pow(d[25],2) + 2*d[11]*d[20]*d[26] + 2*d[14]*d[23]*d[26] + 2*d[15]*d[24]*d[26] + 2*d[16]*d[25]*d[26];
				coeffs[265] = std::pow(d[26],3);
				coeffs[266] = -d[8]*std::pow(d[18],2) - d[8]*std::pow(d[19],2) + 2*d[6]*d[18]*d[20] + 2*d[7]*d[19]*d[20] - d[8]*std::pow(d[21],2) - d[8]*std::pow(d[22],2) + 2*d[6]*d[21]*d[23] + 2*d[7]*d[22]*d[23] + 2*d[2]*d[18]*d[24] + 2*d[0]*d[20]*d[24] + 2*d[5]*d[21]*d[24] + 2*d[3]*d[23]*d[24] + 2*d[2]*d[19]*d[25] + 2*d[1]*d[20]*d[25] + 2*d[5]*d[22]*d[25] + 2*d[4]*d[23]*d[25] - 2*d[0]*d[18]*d[26] - 2*d[1]*d[19]*d[26] - 2*d[3]*d[21]*d[26] - 2*d[4]*d[22]*d[26];
				coeffs[267] = -d[17]*std::pow(d[18],2) - d[17]*std::pow(d[19],2) + 2*d[15]*d[18]*d[20] + 2*d[16]*d[19]*d[20] - d[17]*std::pow(d[21],2) - d[17]*std::pow(d[22],2) + 2*d[15]*d[21]*d[23] + 2*d[16]*d[22]*d[23] + 2*d[11]*d[18]*d[24] + 2*d[9]*d[20]*d[24] + 2*d[14]*d[21]*d[24] + 2*d[12]*d[23]*d[24] + 2*d[11]*d[19]*d[25] + 2*d[10]*d[20]*d[25] + 2*d[14]*d[22]*d[25] + 2*d[13]*d[23]*d[25] - 2*d[9]*d[18]*d[26] - 2*d[10]*d[19]*d[26] - 2*d[12]*d[21]*d[26] - 2*d[13]*d[22]*d[26];
				coeffs[268] = std::pow(d[20],2)*d[26] + std::pow(d[23],2)*d[26] + std::pow(d[24],2)*d[26] + std::pow(d[25],2)*d[26];
				coeffs[269] = 2*d[18]*d[20]*d[24] + 2*d[21]*d[23]*d[24] + 2*d[19]*d[20]*d[25] + 2*d[22]*d[23]*d[25] - std::pow(d[18],2)*d[26] - std::pow(d[19],2)*d[26] - std::pow(d[21],2)*d[26] - std::pow(d[22],2)*d[26];
				coeffs[270] = -d[2]*d[4]*d[6] + d[1]*d[5]*d[6] + d[2]*d[3]*d[7] - d[0]*d[5]*d[7] - d[1]*d[3]*d[8] + d[0]*d[4]*d[8];
				coeffs[271] = -d[5]*d[7]*d[9] + d[4]*d[8]*d[9] + d[5]*d[6]*d[10] - d[3]*d[8]*d[10] - d[4]*d[6]*d[11] + d[3]*d[7]*d[11] + d[2]*d[7]*d[12] - d[1]*d[8]*d[12] - d[2]*d[6]*d[13] + d[0]*d[8]*d[13] + d[1]*d[6]*d[14] - d[0]*d[7]*d[14] - d[2]*d[4]*d[15] + d[1]*d[5]*d[15] + d[2]*d[3]*d[16] - d[0]*d[5]*d[16] - d[1]*d[3]*d[17] + d[0]*d[4]*d[17];
				coeffs[272] = -d[8]*d[10]*d[12] + d[7]*d[11]*d[12] + d[8]*d[9]*d[13] - d[6]*d[11]*d[13] - d[7]*d[9]*d[14] + d[6]*d[10]*d[14] + d[5]*d[10]*d[15] - d[4]*d[11]*d[15] - d[2]*d[13]*d[15] + d[1]*d[14]*d[15] - d[5]*d[9]*d[16] + d[3]*d[11]*d[16] + d[2]*d[12]*d[16] - d[0]*d[14]*d[16] + d[4]*d[9]*d[17] - d[3]*d[10]*d[17] - d[1]*d[12]*d[17] + d[0]*d[13]*d[17];
				coeffs[273] = -d[11]*d[13]*d[15] + d[10]*d[14]*d[15] + d[11]*d[12]*d[16] - d[9]*d[14]*d[16] - d[10]*d[12]*d[17] + d[9]*d[13]*d[17];
				coeffs[274] = -d[5]*d[7]*d[18] + d[4]*d[8]*d[18] + d[5]*d[6]*d[19] - d[3]*d[8]*d[19] - d[4]*d[6]*d[20] + d[3]*d[7]*d[20] + d[2]*d[7]*d[21] - d[1]*d[8]*d[21] - d[2]*d[6]*d[22] + d[0]*d[8]*d[22] + d[1]*d[6]*d[23] - d[0]*d[7]*d[23] - d[2]*d[4]*d[24] + d[1]*d[5]*d[24] + d[2]*d[3]*d[25] - d[0]*d[5]*d[25] - d[1]*d[3]*d[26] + d[0]*d[4]*d[26];
				coeffs[275] = d[8]*d[13]*d[18] - d[7]*d[14]*d[18] - d[5]*d[16]*d[18] + d[4]*d[17]*d[18] - d[8]*d[12]*d[19] + d[6]*d[14]*d[19] + d[5]*d[15]*d[19] - d[3]*d[17]*d[19] + d[7]*d[12]*d[20] - d[6]*d[13]*d[20] - d[4]*d[15]*d[20] + d[3]*d[16]*d[20] - d[8]*d[10]*d[21] + d[7]*d[11]*d[21] + d[2]*d[16]*d[21] - d[1]*d[17]*d[21] + d[8]*d[9]*d[22] - d[6]*d[11]*d[22] - d[2]*d[15]*d[22] + d[0]*d[17]*d[22] - d[7]*d[9]*d[23] + d[6]*d[10]*d[23] + d[1]*d[15]*d[23] - d[0]*d[16]*d[23] + d[5]*d[10]*d[24] - d[4]*d[11]*d[24] - d[2]*d[13]*d[24] + d[1]*d[14]*d[24] - d[5]*d[9]*d[25] + d[3]*d[11]*d[25] + d[2]*d[12]*d[25] - d[0]*d[14]*d[25] + d[4]*d[9]*d[26] - d[3]*d[10]*d[26] - d[1]*d[12]*d[26] + d[0]*d[13]*d[26];
				coeffs[276] = -d[14]*d[16]*d[18] + d[13]*d[17]*d[18] + d[14]*d[15]*d[19] - d[12]*d[17]*d[19] - d[13]*d[15]*d[20] + d[12]*d[16]*d[20] + d[11]*d[16]*d[21] - d[10]*d[17]*d[21] - d[11]*d[15]*d[22] + d[9]*d[17]*d[22] + d[10]*d[15]*d[23] - d[9]*d[16]*d[23] - d[11]*d[13]*d[24] + d[10]*d[14]*d[24] + d[11]*d[12]*d[25] - d[9]*d[14]*d[25] - d[10]*d[12]*d[26] + d[9]*d[13]*d[26];
				coeffs[277] = -d[8]*d[19]*d[21] + d[7]*d[20]*d[21] + d[8]*d[18]*d[22] - d[6]*d[20]*d[22] - d[7]*d[18]*d[23] + d[6]*d[19]*d[23] + d[5]*d[19]*d[24] - d[4]*d[20]*d[24] - d[2]*d[22]*d[24] + d[1]*d[23]*d[24] - d[5]*d[18]*d[25] + d[3]*d[20]*d[25] + d[2]*d[21]*d[25] - d[0]*d[23]*d[25] + d[4]*d[18]*d[26] - d[3]*d[19]*d[26] - d[1]*d[21]*d[26] + d[0]*d[22]*d[26];
				coeffs[278] = -d[17]*d[19]*d[21] + d[16]*d[20]*d[21] + d[17]*d[18]*d[22] - d[15]*d[20]*d[22] - d[16]*d[18]*d[23] + d[15]*d[19]*d[23] + d[14]*d[19]*d[24] - d[13]*d[20]*d[24] - d[11]*d[22]*d[24] + d[10]*d[23]*d[24] - d[14]*d[18]*d[25] + d[12]*d[20]*d[25] + d[11]*d[21]*d[25] - d[9]*d[23]*d[25] + d[13]*d[18]*d[26] - d[12]*d[19]*d[26] - d[10]*d[21]*d[26] + d[9]*d[22]*d[26];
				coeffs[279] = -d[20]*d[22]*d[24] + d[19]*d[23]*d[24] + d[20]*d[21]*d[25] - d[18]*d[23]*d[25] - d[19]*d[21]*d[26] + d[18]*d[22]*d[26];

				static const int coeffs_ind[] = {0,30,60,90,120,150,180,210,240,1,31,61,91,121,151,181,211,241,2,32,62,92,122,152,182,212,242,4,34,64,30,0,60,94,124,154,90,120,150,184,214,180,210,240,244,270,5,35,65,31,
				1,61,95,125,155,91,121,151,185,215,181,211,241,245,271,6,36,66,32,2,62,96,126,156,92,122,152,186,216,182,212,242,246,272,7,37,67,33,3,63,97,127,157,93,123,153,187,217,183,213,
				243,247,273,8,38,68,98,128,158,188,218,248,9,39,69,99,129,159,189,219,249,11,41,71,34,4,64,101,131,161,90,94,60,120,124,154,191,221,0,180,184,150,210,214,30,240,244,251,270,12,
				42,72,35,5,65,102,132,162,91,95,61,121,125,155,192,222,1,181,185,151,211,215,31,241,245,252,271,13,43,73,36,6,66,103,133,163,92,96,62,122,126,156,193,223,2,182,186,152,212,216,
				32,242,246,253,272,14,44,74,37,7,67,104,134,164,93,97,63,123,127,157,194,224,3,183,187,153,213,217,33,243,247,254,273,15,45,75,38,8,68,105,135,165,98,128,158,195,225,188,218,248,
				255,274,16,46,76,39,9,69,106,136,166,99,129,159,196,226,189,219,249,256,275,17,47,77,40,10,70,107,137,167,100,130,160,197,227,190,220,250,257,276,18,48,78,108,138,168,198,228,258,41,
				11,71,94,101,64,124,131,161,4,184,191,154,214,221,34,244,251,270,42,12,72,95,102,65,125,132,162,5,185,192,155,215,222,35,245,252,271,20,50,80,45,15,75,110,140,170,98,105,68,128,
				135,165,200,230,8,188,195,158,218,225,38,248,255,260,274,23,53,83,48,18,78,113,143,173,108,138,168,203,233,198,228,258,263,277,101,71,131,11,191,161,221,41,251,270,50,20,80,105,110,75,
				135,140,170,15,195,200,165,225,230,45,255,260,274,102,72,132,12,192,162,222,42,252,271,103,73,133,13,193,163,223,43,253,272,43,13,73,96,103,66,126,133,163,6,186,193,156,216,223,36,246,
				253,272,21,51,81,46,16,76,111,141,171,99,106,69,129,136,166,201,231,9,189,196,159,219,226,39,249,256,261,275,104,74,134,14,194,164,224,44,254,273,44,14,74,97,104,67,127,134,164,7,
				187,194,157,217,224,37,247,254,273,22,52,82,47,17,77,112,142,172,100,107,70,130,137,167,202,232,10,190,197,160,220,227,40,250,257,262,276,24,54,84,49,19,79,114,144,174,109,139,169,204,
				234,199,229,259,264,278,119,89,149,29,209,179,239,59,269,279,116,86,146,26,206,176,236,56,266,277,110,80,140,20,200,170,230,50,260,274,111,81,141,21,201,171,231,51,261,275,51,21,81,106,
				111,76,136,141,171,16,196,201,166,226,231,46,256,261,275,56,26,86,113,116,83,143,146,176,23,203,206,173,233,236,53,263,266,277,26,56,86,53,23,83,116,146,176,108,113,78,138,143,173,206,
				236,18,198,203,168,228,233,48,258,263,266,277,117,87,147,27,207,177,237,57,267,278,112,82,142,22,202,172,232,52,262,276,52,22,82,107,112,77,137,142,172,17,197,202,167,227,232,47,257,262,
				276,57,27,87,114,117,84,144,147,177,24,204,207,174,234,237,54,264,267,278,27,57,87,54,24,84,117,147,177,109,114,79,139,144,174,207,237,19,199,204,169,229,234,49,259,264,267,278,59,29,
				89,118,119,88,148,149,179,28,208,209,178,238,239,58,268,269,279,29,59,89,58,28,88,119,149,179,115,118,85,145,148,178,209,239,25,205,208,175,235,238,55,265,268,269,279,28,58,88,55,25,
				85,118,148,178,115,145,175,208,238,205,235,265,268,279};

				static const int C_ind[] = {0,1,2,6,7,8,15,16,26,31,32,33,37,38,39,46,47,57,62,63,64,68,69,70,77,78,88,93,94,95,96,97,98,99,100,101,103,106,107,108,109,112,115,118,119,123,124,125,126,127,
				128,129,130,131,132,134,137,138,139,140,143,146,149,150,154,155,156,157,158,159,160,161,162,163,165,168,169,170,171,174,177,180,181,185,186,187,188,189,190,191,192,193,194,196,199,200,201,202,205,208,
				211,212,216,217,218,219,223,224,225,232,233,243,248,249,250,254,255,256,263,264,274,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,308,310,
				311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,339,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,
				364,365,366,367,370,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,401,403,404,405,406,407,408,409,410,411,413,416,417,418,419,422,425,428,
				429,433,434,435,436,437,438,439,440,441,442,444,447,448,449,450,453,456,459,460,464,465,466,467,468,469,470,471,472,473,475,478,479,480,481,484,487,490,491,495,496,497,498,502,503,504,511,512,522,530,
				531,532,536,537,538,539,540,541,544,545,546,547,548,549,550,551,552,555,561,562,563,567,568,569,570,571,572,575,576,577,578,579,580,581,582,583,586,589,590,591,592,593,594,595,596,597,598,599,600,601,
				602,603,604,605,606,607,608,609,610,611,612,613,614,615,618,620,621,622,623,624,625,626,627,628,630,633,634,635,636,639,642,645,646,650,660,662,663,668,669,671,672,674,675,678,685,686,687,691,692,693,
				694,695,696,699,700,701,702,703,704,705,706,707,710,722,724,725,730,731,733,734,736,737,740,753,755,756,761,762,764,765,767,768,771,778,779,780,784,785,786,787,788,789,792,793,794,795,796,797,798,799,
				800,803,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,835,846,848,849,854,855,857,858,860,861,864,871,872,873,877,878,879,880,881,882,885,
				886,887,888,889,890,891,892,893,896,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,928,930,931,932,933,934,935,936,937,938,940,943,944,945,
				946,949,952,955,956,960,970,972,973,978,979,981,982,984,985,988,1001,1003,1004,1009,1010,1012,1013,1015,1016,1019,1032,1034,1035,1040,1041,1043,1044,1046,1047,1050,1063,1065,1066,1071,1072,1074,1075,1077,1078,1081,1088,1089,1090,1094,
				1095,1096,1097,1098,1099,1102,1103,1104,1105,1106,1107,1108,1109,1110,1113,1119,1120,1121,1125,1126,1127,1128,1129,1130,1133,1134,1135,1136,1137,1138,1139,1140,1141,1144,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,
				1163,1164,1165,1166,1167,1168,1169,1170,1171,1172,1173,1176,1187,1189,1190,1195,1196,1198,1199,1201,1202,1205,1218,1220,1221,1226,1227,1229,1230,1232,1233,1236,1243,1244,1245,1249,1250,1251,1252,1253,1254,1257,1258,1259,1260,1261,1262,1263,1264,1265,
				1268,1274,1275,1276,1280,1281,1282,1283,1284,1285,1288,1289,1290,1291,1292,1293,1294,1295,1296,1299,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1320,1321,1322,1323,1324,1325,1326,1327,1328,1331,1336,1337,
				1338,1342,1343,1344,1345,1346,1347,1350,1351,1352,1353,1354,1355,1356,1357,1358,1361,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1393,1395,1396,1397,1398,1399,
				1400,1401,1402,1403,1405,1408,1409,1410,1411,1414,1417,1420,1421,1425};

				MatrixXd C = MatrixXd::Zero(31,46);
				for (int i = 0; i < 814; i++) {
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(31);
				MatrixXd C1 = C.rightCols(15);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(23, 15);
				RR << -C12.bottomRows(8), MatrixXd::Identity(15, 15);

				static const int AM_ind[] = {15,11,0,1,2,12,3,16,4,5,17,6,18,19,7};
				MatrixXd AM(15, 15);
				for (int i = 0; i < 15; i++) {
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).replicate(15, 1)).eval();

				MatrixXcd sols(3, 15);
				sols.row(0) = V.row(1);
				sols.row(1) = D.transpose();
				sols.row(2) = V.row(12);
				return sols;
			}
		}
	}
}