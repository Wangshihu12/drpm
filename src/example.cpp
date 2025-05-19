#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include "data.h"
#include "degeneracy.h"

/**
 * 计算黑塞矩阵（Hessian matrix）
 * 该函数基于点云数据、法向量和权重计算6x6的黑塞矩阵
 * 
 * @param points 输入的三维点集合
 * @param normals 与点对应的法向量集合
 * @param weights 每个点的权重
 * @return 计算得到的6x6黑塞矩阵
 */
Eigen::Matrix<double, 6, 6> ComputeHessian(const degeneracy::VectorVector3<double>& points, const degeneracy::VectorVector3<double>& normals, const std::vector<double>& weights) {
  // 获取点的数量
  const size_t nPoints = points.size();
  // 初始化6x6的黑塞矩阵，全部元素为0
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(6, 6);
  // 遍历所有点
  for (size_t i = 0; i < nPoints; i++) {
    // 获取当前点
    const Eigen::Vector3d point = points[i];
    // 获取当前点对应的法向量
    const Eigen::Vector3d normal = normals[i];
    // 计算点与法向量的叉积（cross product）
    const Eigen::Vector3d pxn = point.cross(normal);
    // 计算权重的平方根
    const double w = std::sqrt(weights[i]);
    // 创建一个6维向量
    Eigen::Matrix<double, 6, 1> v;
    // 前3个元素设置为加权后的叉积
    v.head(3) = w * pxn;
    // 后3个元素设置为加权后的法向量
    v.tail(3) = w * normal;
    // 通过向量的外积更新黑塞矩阵
    H += v * v.transpose();
  }
  // 返回计算得到的黑塞矩阵
  return H;
}

/**
 * 创建等方差协方差矩阵集合
 * 
 * @param N 需要创建的协方差矩阵数量
 * @param stdev 标准差参数，用于计算协方差值
 * @return 包含N个等方差协方差矩阵的向量
 */
degeneracy::VectorMatrix3<double> GetIsotropicCovariances(const size_t& N, const double stdev) {
  // 创建3x3矩阵的向量容器
  degeneracy::VectorMatrix3<double> covariances;
  // 预分配N个元素的内存空间，提高性能
  covariances.reserve(N);
  // 循环创建N个协方差矩阵
  for (size_t i = 0; i < N; i++) {
    // 添加一个3x3的等方差协方差矩阵
    // Eigen::Matrix3d::Identity()创建单位矩阵
    // 乘以标准差的平方(方差)使其成为等方差协方差矩阵
    covariances.push_back(Eigen::Matrix3d::Identity() * std::pow(stdev, 2));
  }
  // 返回创建的协方差矩阵集合
  return covariances;
}

/**
 * 主函数 - 演示点云退化分析流程
 * 处理点云数据并分析其几何特性以检测退化情况
 */
int main() {
  // 注意：点、法向量和协方差必须在同一参考坐标系中表示
  // 对于黑塞矩阵的条件数计算，最好使用激光雷达坐标系（而非世界坐标系）
  // 从data命名空间加载预定义的点云数据
  const auto points = data::points;                   // 加载3D点云数据
  const auto normals = data::normals;                 // 加载对应的法向量
  const auto weights_squared = data::weights_squared; // 加载权重的平方值
  // 为法向量创建等方差协方差矩阵
  const auto normal_covariances = GetIsotropicCovariances(data::normals.size(), data::stdev_normals);

  // 计算黑塞矩阵H
  const auto H = ComputeHessian(points, normals, weights_squared);
  // 使用自伴随特征值求解器分解黑塞矩阵（适用于对称矩阵）
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(H);

  // 提取特征向量和特征值
  const auto eigenvectors = eigensolver.eigenvectors(); // 获取特征向量
  const auto eigenvalues = eigensolver.eigenvalues();   // 获取特征值

  // 初始化噪声估计的变量
  Eigen::Matrix<double, 6, 6> noise_mean;    // 噪声平均值（6x6矩阵）
  Eigen::Matrix<double, 6, 1> noise_variance; // 噪声方差（6维向量）
  const double snr_factor = 10.0;            // 信噪比因子，用于计算退化概率

  // 计算噪声估计，结果通过std::tie存储到之前声明的变量中
  std::tie(noise_mean, noise_variance) = degeneracy::ComputeNoiseEstimate<double, double>(
      points, normals, weights_squared, normal_covariances, eigenvectors, data::stdev_points);
  
  // 计算非退化概率 - 评估沿每个特征向量方向的信号强度相对于噪声的可靠性
  Eigen::Matrix<double, 6, 1> non_degeneracy_probabilities = degeneracy::ComputeSignalToNoiseProbabilities<double>(
      H, noise_mean, noise_variance, eigenvectors, snr_factor);

  // 输出结果
  std::cout << "The non-degeneracy probabilities are: " << std::endl;
  std::cout << non_degeneracy_probabilities.transpose() << std::endl;

  std::cout << "For the eigenvectors of the Hessian: " << std::endl;
  std::cout << eigenvectors << std::endl;

  // 以下代码展示了如何使用概率解决方程组，但当前被注释掉
  // // 创建一个示例右侧向量rhs = Jtb（全零）
  // const Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero(6, 1);
  // // 使用信噪比概率求解方程
  // const auto estimate = degeneracy::SolveWithSnrProbabilities(eigenvectors, eigenvalues, rhs, non_degeneracy_probabilities);

  return 0;
}
