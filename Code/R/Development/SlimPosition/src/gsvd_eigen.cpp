#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp ;
using namespace Eigen ;

typedef std::tuple<MatrixXd, MatrixXd, VectorXd> basic_svd ;

void destructive_threshold_matrix(MatrixXd& m, double tol){
  for(int i = 0; i < m.rows(); ++i)
    for(int j = 0; j < m.cols(); ++j)
      if(abs(m(i,j)) < tol)
	m(i,j) = 0;
}


basic_svd tolerance_svd_cpp(Eigen::MatrixXd m, double tol, int k){
  int n = m.rows();
  int p = m.cols();
  BDCSVD<MatrixXd> solver(m, ComputeThinU | ComputeThinV);
  //JacobiSVD<MatrixXd> solver(m, ComputeThinU | ComputeThinV);
  
  VectorXd d = solver.singularValues();

  //Compute the rank given tolerance
  int rank = 0;
  for(int i = 0; i < d.size(); ++i){
    if(d(i) * d(i) <  tol) break;
    ++rank;
  }

  //// Prints for debug
  // Rcout << u.rows() << " " << solver.matrixU().cols() << '\n';
  // Rcout << v.rows() << " " << solver.matrixV().cols() << '\n';
  // Rcout << rank << '\n';
  // Rcout << d.head(10) << "\n";

  // Check user supplied max components, take min of it and rank
  if(k < rank)
    rank = k;
  
  MatrixXd lru = solver.matrixU().block(0,0,n,rank);
  MatrixXd lrv = solver.matrixV().block(0,0,p,rank);
  VectorXd lrd = d.head(rank);

  if(!lrd.allFinite())
    stop("Matrix is not diagonalizable, maybe try centering/scaling");

  // Zero lru and lrv when small (probably not necessary, but quick)
  destructive_threshold_matrix(lru, tol);
  destructive_threshold_matrix(lrv, tol);

  return(std::make_tuple(lru, lrv, lrd));
}


// [[Rcpp::export]]
Rcpp::List tolerance_svd(Eigen::MatrixXd m, double tol, int k){
    std::tuple<MatrixXd, MatrixXd, VectorXd> svd;
    svd = tolerance_svd_cpp(m, tol, k);

    return(List::create(_["u"] = std::get<0>(svd)
			, _["v"] = std::get<1>(svd)
			, _["d"] = std::get<2>(svd)));
  }

Eigen::MatrixXd matrix_power(basic_svd svd, double power, double tol){
    MatrixXd lru = std::get<0>(svd);
    MatrixXd lrv = std::get<1>(svd);
    VectorXd lrd = std::get<2>(svd);
    int rank = lrd.size();
    // Exponentiate singular values
    for(int j = 0; j < rank; ++j)
      lrd(j) = pow(lrd(j), power);

  
    // Put it back together
    MatrixXd res = lru * lrd.asDiagonal() * lrv.transpose();
    destructive_threshold_matrix(res, tol);

    return(res);
}

// [[Rcpp::export]]
Eigen::MatrixXd matrix_power(Eigen::MatrixXd m, double power, double tol, int k){

  // Compute the tolerance SVD
  basic_svd svd = tolerance_svd_cpp(m, tol, k);

  return(matrix_power(svd, power, tol));
}

// [[Rcpp::export]]
Rcpp::List matrix_powers(Eigen::MatrixXd m, std::vector<double> powers, double tol, int k){

  // Compute the tolerance SVD
  basic_svd svd = tolerance_svd_cpp(m, tol, k);
  Rcpp::List res(powers.size());

  for(int i = 0; i < powers.size(); ++i)
    res(i) = matrix_power(svd, powers[i], tol);

  return(res);
}



VectorXd vector_power(VectorXd v, double power, double tol, int k){
  int rank = 0;
  for(int i = 0; i < v.size(); ++i){
    v(i) = pow(v(i), power);
    if(v(i) < tol) break;
    
    ++rank;
  }

  if(k < rank) rank = k;

  return(v.head(rank));
}


// [[Rcpp::export]]
List gsvd_eig(Eigen::MatrixXd data
	      , Eigen::MatrixXd left_weights
	      , Eigen::MatrixXd right_weights
	      , double tol
	      , int nv
	      , int k){

  // Compute powers of the left weights
  int l_comps = std::min(left_weights.rows(), left_weights.cols());
  std::tuple<MatrixXd, MatrixXd, VectorXd> lw_svd;
  lw_svd = tolerance_svd_cpp(left_weights, tol, l_comps);
  MatrixXd lw_sqrt = matrix_power(lw_svd, .5, tol);
  MatrixXd lw_sqrt_inv = matrix_power(lw_svd, -.5, tol);

  // Compute powers of the right weights
  int r_comps = std::min(right_weights.rows(), right_weights.cols());
  std::tuple<MatrixXd, MatrixXd, VectorXd> rw_svd;
  rw_svd = tolerance_svd_cpp(right_weights, tol, r_comps);
  MatrixXd rw_sqrt = matrix_power(rw_svd, .5, tol);
  MatrixXd rw_sqrt_inv = matrix_power(rw_svd, -.5, tol);

  // Multiply half weights in
  data = lw_sqrt * data * rw_sqrt;

  // Compute svd
  std::tuple<MatrixXd, MatrixXd, VectorXd> svd;
  svd = tolerance_svd_cpp(data, tol, nv);

  MatrixXd u = std::get<0>(svd);
  MatrixXd v = std::get<1>(svd);
  VectorXd d = std::get<2>(svd);

  VectorXd tau = d.array().pow(2).matrix();
  tau /= tau.array().sum();

  MatrixXd p = lw_sqrt_inv * u;
  MatrixXd fi = left_weights * p * d.asDiagonal();

  MatrixXd q = rw_sqrt_inv * v;
  MatrixXd fj = right_weights * q * d.asDiagonal();

  return(List::create(_["fi"] = fi
		      , _["fj"] = fj
		      , _["p"] = p
		      , _["q"] = q
		      , _["u"] = u
		      , _["v"] = v
		      , _["d"] = d
		      , _["tau"] = tau));
}
