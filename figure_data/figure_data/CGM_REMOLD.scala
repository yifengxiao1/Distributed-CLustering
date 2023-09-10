/**
  * Created by Admin on 2017/12/24.
  *
  * */

import java.io._
import java.util.concurrent.{Callable, ExecutorService, Executors, FutureTask}

import breeze.linalg.{*, norm, Axis, DenseMatrix, DenseVector, diag, max, min, pinv, svd}
import breeze.numerics._
import breeze.stats.mean
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.linalg.DenseVector
//import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, WrappedArray}
import scala.math.{exp, pow, sqrt, log}



object CGM_REMOLD {


  val numeric_lower_bound = 1e-20
  //  def euclideanDis(x_arr: Array[Double], y_arr: Array[Double]): Double = {
  //    var res = 0.0
  //    var index = 0
  //    while (index < x_arr.length) {
  //      val tmp = x_arr(index) - y_arr(index)
  //      res += pow(tmp, 2)
  //      index += 1
  //    }
  //    return sqrt(res)
  //  }
  //  def euclideanDis(x_arr: org.apache.spark.ml.linalg.Vector, y_arr: org.apache.spark.ml.linalg.Vector): Double = {
  //    return org.apache.spark.ml.linalg.Vectors.sqdist(x_arr, y_arr)
  //  }

  def euclideanDis(x_arr: DenseVector[Double], y_arr: DenseVector[Double]): Double = {
    return sqrt(squaredEuclideanDis(x_arr, y_arr))
  }

  def squaredEuclideanDis(x_arr: DenseVector[Double], y_arr: DenseVector[Double]): Double = {
    return breeze.linalg.squaredDistance(x_arr, y_arr)
  }


  class UnionFindSet(n: Int) {
    val unionFindSet: Array[Int] = new Array(n)
    for (i <- 0 until n) {
      this.unionFindSet(i) = i
    }

    def find(i: Int): Int = {
      if (i != this.unionFindSet(i)) {
        this.unionFindSet(i) = find(unionFindSet(i))
      }
      return this.unionFindSet(i)
    }

    def union(x: Int, y: Int) = {
      val rootX = find(x)
      val rootY = find(y)
      if (rootX != rootY) {
        this.unionFindSet(rootY) = rootX
      }
    }
  }

  def HeapAdjust(array: Array[(Int, Double)], t_parent: Int, length: Int): Unit = {
    val temp = array(t_parent)
    var parent = t_parent
    var child = 2 * parent + 1
    while (child < length) {
      if (child + 1 < length && array(child)._2 > array(child + 1)._2) {
        child += 1
      }
      if (temp._2 <= array(child)._2) {
        array(parent) = temp
        return
      }
      array(parent) = array(child)
      parent = child
      child = 2 * child + 1
    }
    array(parent) = temp
  }

  def buildHeap(array: Array[(Int, Double)]): Unit = {
    var i = array.length / 2
    val length = array.length
    while (i >= 0) {
      HeapAdjust(array, i, length)
      i -= 1
    }
  }

  def topK(array: Array[(Int, Double)], K: Int): Array[(Int, Double)] = {
    var k = K
    val listTmp = new Array[Tuple2[Int, Double]](k)
    val length = array.length
    listTmp(0) = array(0)
    var i = length - 1
    while (i > 0 && k > 1) {
      val temp = array(i)
      array(i) = array(0)
      array(0) = temp
      HeapAdjust(array, 0, i)
      listTmp(length - i) = array(0)
      i -= 1
      k -= 1
    }
    return listTmp
  }

  def removefirst_orlast(str: String): String = {
    str.substring(1, str.length - 1)
  }

  def cal_NMI(ori_label1: Array[Int], ori_label2: Array[Int], ori_n: Int): (Double, Int) = {

    val label1 = new ArrayBuffer[Int]()
    val label2 = new ArrayBuffer[Int]()

    for (i <- 0 until ori_n) {
      if (ori_label2(i) != -1) {
        label1 += ori_label1(i)
        label2 += ori_label2(i)
      }
    }
    val n = label1.length
    val index_map1 = new mutable.HashMap[Int, Int]
    var count = 1
    for (i <- 0 until n) {
      if (!index_map1.contains(label1(i))) {
        index_map1(label1(i)) = count
        count += 1
      }
      label1(i) = index_map1(label1(i))
    }

    //    val index_map2 = new mutable.HashMap[Int, Int]
    //    count = 1
    //    for (i <- 0 until n) {
    //      if (!index_map1.contains(label1(i))) {
    //        index_map1(label1(i)) = count
    //        count += 1
    //      }
    //      label1(i) = index_map1(label1(i))
    //    }


    val X = label1.max
    val Y = label2.max
    val cnt = DenseMatrix.zeros[Double](X, Y)
    val cnt1 = DenseVector.zeros[Double](X)
    val cnt2 = DenseVector.zeros[Double](Y)
    for (i <- 0 until n) {
      cnt(label1(i) - 1, label2(i) - 1) += 1
      cnt1(label1(i) - 1) += 1
      cnt2(label2(i) - 1) += 1
    }
    var MI = 0.0
    for (i <- 0 until X) {
      for (j <- 0 until Y) {
        if (cnt(i, j) > 0) {
          MI += (cnt(i, j) / n) * log((cnt(i, j) * n) / (cnt1(i) * cnt2(j)))
        }
      }
    }
    var H1 = 0.0
    var H2 = 0.0
    for (i <- 0 until X) {
      if (cnt1(i) > 0) {
        H1 -= (cnt1(i) / n) * log(cnt1(i) / n)
      }
    }

    for (i <- 0 until Y) {
      if (cnt2(i) > 0) {
        H2 -= (cnt2(i) / n) * log(cnt2(i) / n)
      }
    }
    return (MI / ((H1 + H2) / 2), label1.max)
  }

  def cal_purity(ori_label1: Array[Int], ori_label2: Array[Int], ori_n: Int): (Double, Int) = {
    // label1 = predict label2 = real_label
    val label1 = new ArrayBuffer[Int]()
    val label2 = new ArrayBuffer[Int]()

    for (i <- 0 until ori_n) {
      if (ori_label2(i) != -1) {
        label1 += ori_label1(i)
        label2 += ori_label2(i)
      }
    }
    val n = label1.length
    val index_map = new mutable.HashMap[Int, Int]
    var count = 1
    for (i <- 0 until n) {
      if (!index_map.contains(label1(i))) {
        index_map(label1(i)) = count
        count += 1
      }
      label1(i) = index_map(label1(i))
    }
    val X = label1.max
    val Y = label2.max
    val cnt = DenseMatrix.zeros[Double](X, Y)
    val cnt1 = DenseVector.zeros[Double](X)
    val cnt2 = DenseVector.zeros[Double](Y)
    for (i <- 0 until n) {
      cnt(label1(i) - 1, label2(i) - 1) += 1
      cnt1(label1(i) - 1) += 1
      cnt2(label2(i) - 1) += 1
    }
    var itersection_count = 0.0
    for (i <- 0 until X) {
      itersection_count += max(cnt(i, ::))
    }
    return (itersection_count / n, label1.max)
  }

  class REMOLD_GuassianModel(d: Int, partitionIndex: Int, modelIndex: Int) extends Serializable {
    var mean: breeze.linalg.DenseVector[Double] = null
    var inverseMatrix: breeze.linalg.DenseMatrix[Double] = new breeze.linalg.DenseMatrix[Double](d, d)
    var logSqrtDet: Double = 0
    val partitionId = partitionIndex
    val modelId = modelIndex
    var datasize: Double = 0
    var r: Int = 0
  }

  class CGM_GuassianModel(d: Int, partitionIndex: Int, modelIndex: Int) extends Serializable {
    var mean: DenseVector[Double] = null
    var dig_invMatrix: breeze.linalg.DenseVector[Double] = null
    var P_Matrix: breeze.linalg.DenseMatrix[Double] = null
    var sig: Double = 0
    var logSqrtDet: Double = 0
    val partitionId = partitionIndex
    val modelId = modelIndex
    var datasize: Double = 0
  }

  class GuassianData(d: Int, partitionIndex: Int, modelIndex: Int) extends Serializable {
    //data array
    var data: scala.collection.mutable.ArrayBuffer[DenseVector[Double]] = scala.collection.mutable.ArrayBuffer[DenseVector[Double]]()
    //var datamatrix: DenseMatrix[Double] = null
    val partitionId: Int = partitionIndex
    val modelId: Int = modelIndex
    val label: scala.collection.mutable.ArrayBuffer[Double] = scala.collection.mutable.ArrayBuffer[Double]()

    def add(cur_data: DenseVector[Double], cur_label: Double) = {
      data += cur_data
      label += cur_label
    }
  }


  //带数据的DataWithLabelPredict
  // 样本量，标签数组，预测数组
  class LabelPredict(SampleSize: Int, Labels: Array[Double], Predict: Int) extends Serializable {
    val LabelPredict = new Array[Array[Int]](SampleSize)
    for (i <- 0 until SampleSize) {
      LabelPredict(i) = new Array[Int](2)
      LabelPredict(i)(0) = math.round(Labels(i)).toInt
      LabelPredict(i)(1) = math.round(Predict).toInt
    }
  }

  //  不带数据的DataWithLabelPredict
  //  class DataWithLabelPredict(Labels: Array[Double], Predict: Int) extends Serializable {
  //    val dataLabelPredict = new Array[Array[Double]](Labels.length)
  //    for (i <- 0 until Labels.length) {
  //      dataLabelPredict(i) = new Array[Double](2)
  //      dataLabelPredict(i)(0) = Labels(i)
  //      dataLabelPredict(i)(1) = Predict
  //    }
  //  }


//  class LSHPartitioner(numParts: Int) extends org.apache.spark.Partitioner {
//    override def numPartitions: Int = numParts
//
//    override def getPartition(key: Any): Int = {
//      val key_ = key.asInstanceOf[Int]
//      key_
//    }
//
//    override def equals(other: Any): Boolean = other match {
//      case lsh: LSHPartitioner =>
//        lsh.numPartitions == numPartitions
//      case _ =>
//        false
//    }
//
//    override def hashCode: Int = numPartitions
//  }

  class LSHPartitioner(numParts: Int, partitionSize: Int) extends org.apache.spark.Partitioner {
    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      val index = key.asInstanceOf[Long].toInt
      val partition_Id = index / partitionSize
      return partition_Id
    }

    override def equals(other: Any): Boolean = other match {
      case lsh: LSHPartitioner =>
        lsh.numPartitions == numPartitions
      case _ =>
        false
    }

    override def hashCode: Int = numPartitions
  }

  def REMOLD_calculateGuassian(alpha: Double, x_index: Int, field_index: Int, y_index: Int, XYX_Matrix: Array[Array[Double]], XYY_Matrix: Array[Array[Double]], arrModel: Array[REMOLD_GuassianModel], xyz_double: Double, dimension: Double): Double = {
    //    return arrModel(field_index).datasize / arrModel(field_index).sqrtdet * exp(-(alpha * alpha * XYX_Matrix(x_index)(field_index) + (1 - alpha) * (1 - alpha) * XYX_Matrix(y_index)(field_index) + XYX_Matrix(field_index)(field_index) -
    //    2 * (1 - alpha) * XYY_Matrix(y_index)(field_index) - 2 * alpha * XYY_Matrix(x_index)(field_index) + 2 * alpha * (1 - alpha) * xyz_double) / 2.0)
    //    var log_ratio = 0.0
    //    log_ratio = -100.0
    return arrModel(field_index).datasize * (-(alpha * alpha * XYX_Matrix(x_index)(field_index) + (1 - alpha) * (1 - alpha) * XYX_Matrix(y_index)(field_index) + XYX_Matrix(field_index)(field_index) -
      2 * (1 - alpha) * XYY_Matrix(y_index)(field_index) - 2 * alpha * XYY_Matrix(x_index)(field_index) + 2 * alpha * (1 - alpha) * xyz_double) / 2.0 - arrModel(field_index).logSqrtDet)

  }

  def REMOLD_computeAllModel(alpha: Double, x_index: Int, y_index: Int, index_set: Array[Int], arrModel: Array[REMOLD_GuassianModel], XYX_Matrix: Array[Array[Double]], XYY_Matrix: Array[Array[Double]], XYZ_arr: mutable.HashMap[Int, Double], dimension: Double): Double = {
    var sum = 0.0
    for (field_index <- index_set) {
      val tmp = REMOLD_calculateGuassian(alpha, x_index, field_index, y_index, XYX_Matrix, XYY_Matrix, arrModel, XYZ_arr(field_index), dimension)
      sum += tmp
    }
    return sum
  }

  def CGM_calculateGuassian(mean_of_log_pow_sig: Double, mean_of_logSqrtDet: Double, alpha: Double, x_index: Int, field_index: Int, y_index: Int, arrModel: Array[CGM_GuassianModel], XY_pri_1: DenseMatrix[Double], XY_pri_2: DenseMatrix[Double]): Double = {
    var XY_pri_1_sum = 0.0
    XY_pri_1_sum += alpha * alpha * XY_pri_1(0, 0)
    XY_pri_1_sum += 2 * alpha * (1 - alpha) * XY_pri_1(0, 1)
    XY_pri_1_sum += -2 * alpha * XY_pri_1(0, 2)
    XY_pri_1_sum += (1 - alpha) * (1 - alpha) * XY_pri_1(1, 1)
    XY_pri_1_sum += -2 * (1 - alpha) * XY_pri_1(1, 2)
    XY_pri_1_sum += XY_pri_1(2, 2)

    var XY_pri_2_sum = 0.0
    XY_pri_2_sum += alpha * alpha * XY_pri_2(0, 0)
    XY_pri_2_sum += 2 * alpha * (1 - alpha) * XY_pri_2(0, 1)
    XY_pri_2_sum += -2 * alpha * XY_pri_2(0, 2)
    XY_pri_2_sum += (1 - alpha) * (1 - alpha) * XY_pri_2(1, 1)
    XY_pri_2_sum += -2 * (1 - alpha) * XY_pri_2(1, 2)
    XY_pri_2_sum += XY_pri_2(2, 2)

    //    val part1 = 1.0 / (pow(arrModel(field_index).sig, (arrModel(field_index).P_Matrix.rows - arrModel(field_index).P_Matrix.cols) / 2) * arrModel(field_index).sqrtdet)
    //    val part2 = exp(-0.5 * XY_pri_1_sum)
    //    val part3 = exp(-0.5 * XY_pri_2_sum / arrModel(field_index).sig)
    //    val ret = arrModel(field_index).datasize * part1 * part2 * part3
    //    return ret
    // var log_ratio = 0.0
    // log_ratio = -100.0
    val tmp = -0.5 * XY_pri_1_sum - (0.5 * XY_pri_2_sum / arrModel(field_index).sig) - (((arrModel(field_index).P_Matrix.rows - arrModel(field_index).P_Matrix.cols) / 2.0) * math.log(arrModel(field_index).sig) - mean_of_log_pow_sig) - (arrModel(field_index).logSqrtDet-mean_of_logSqrtDet)
    //return arrModel(field_index).datasize * exp(log_ratio * log(10) + tmp)
    return arrModel(field_index).datasize * tmp
  }

  def CGM_computeAllModel(mean_of_log_pow_sig: Double, mean_of_logSqrtDet: Double, alpha: Double, x_index: Int, y_index: Int, index_set: Array[Int], arrModel: Array[CGM_GuassianModel], XY_pri_1_map: mutable.HashMap[Int, DenseMatrix[Double]], XY_pri_2_map: mutable.HashMap[Int, DenseMatrix[Double]]): Double = {
    var sum = 0.0
    for (field_index <- index_set) {
      sum += CGM_calculateGuassian(mean_of_log_pow_sig, mean_of_logSqrtDet, alpha, x_index, field_index, y_index, arrModel, XY_pri_1_map(field_index), XY_pri_2_map(field_index))
    }
    return sum
  }

  def CGM_calMixtureDensity(x: DenseVector[Double], arrModel: Array[CGM_GuassianModel]): Double ={
    var sum = 0.0
    for (i <- 0 until arrModel.size){
      val px = arrModel(i).P_Matrix.t*(x-arrModel(i).mean)
      val ppx = arrModel(i).P_Matrix*px
      var tmp = 0.0
      for (j <- 0 until arrModel(i).dig_invMatrix.length)
        tmp += px(j)*px(j)*arrModel(i).dig_invMatrix(j)
      tmp += norm(ppx)/arrModel(i).sig
      sum += arrModel(i).datasize*(-((arrModel(i).P_Matrix.rows - arrModel(i).P_Matrix.cols) / 2.0)*math.log(arrModel(i).sig)
                                    -arrModel(i).logSqrtDet
                                    -0.5*tmp)
    }
    return sum
  }

  def getGuassianData(partitionIndex: Int, iterator: Iterator[(DenseVector[Double], Double)], dimension: Int, ratio: Double, partitionK: Int): Iterator[GuassianData] = {
    val iterList = iterator.toArray
    val partitionSize = iterList.size
    println("partitionSize: "+ partitionSize)
    val local_K = math.min(partitionK, partitionSize)
    //调节K
    // val partitionK = (sqrt(partitionSize) * ratio).toInt
    val index_KNN = new Array[Array[Int]](partitionSize)
    val dis_KNN = new Array[Array[Double]](partitionSize)
    val heap_tmp = new Array[(Int, Double)](partitionSize-1)
    var sig = 0.0
    for (i <- 0 until partitionSize) {
      val x_arr = iterList(i)
      var tmp_i = 0
      for (j <- 0 until partitionSize)
        if(i != j){
          val y_arr = iterList(j)
          heap_tmp(tmp_i) = Tuple2(j, squaredEuclideanDis(x_arr._1, y_arr._1))
          tmp_i += 1
         }
      buildHeap(heap_tmp)
      val topK_index_with_dis = topK(heap_tmp, local_K)
      index_KNN(i) = topK_index_with_dis.map(_._1)
      dis_KNN(i) = topK_index_with_dis.map(x => sqrt(x._2))
      sig += dis_KNN(i)(local_K - 1)
    }
    sig /= partitionSize
    val rho = new Array[Double](partitionSize)

    //KNN Laplace密度
    for (i <- 0 until partitionSize) {
      rho(i) = 0.0
      for (j <- 0 until local_K) {
        rho(i) += exp(-dis_KNN(i)(j) / sig)
      }
    }

    val unionFindSet = new UnionFindSet(partitionSize)
    //并查集union
    for (i <- 0 until partitionSize) {
      var firstFlag: Boolean = true
      for (j <- 0 until local_K if firstFlag) {
        if (rho(index_KNN(i)(j)) > rho(i) || (rho(index_KNN(i)(j)) == rho(i) && (index_KNN(i)(j) > i))) {
          firstFlag = false
          unionFindSet.union(index_KNN(i)(j), i)
        }
      }
    }
    var gaussianModelCount: Int = 0
    val mapIndex = new mutable.HashMap[Long, Int]
    for (i <- 0 until partitionSize) {
      if (unionFindSet.find(i) == i) {
        mapIndex(i) = gaussianModelCount
        gaussianModelCount += 1
      }
    }
    val GuassianArray = new Array[GuassianData](gaussianModelCount)

    for (i <- 0 until gaussianModelCount) {
      GuassianArray(i) = new GuassianData(dimension, partitionIndex, i)
    }

    for (i <- 0 until partitionSize) {
      val guassianIndex = mapIndex(unionFindSet.find(i))
      GuassianArray(guassianIndex).add(iterList(i)._1, iterList(i)._2)
    }
//    for (i <- 0 until gaussianModelCount) {
//      GuassianArray(i).datamatrix = new breeze.linalg.DenseMatrix(GuassianArray(i).data.length, dimension, GuassianArray(i).data.toArray.flatten, 0, dimension, true)
//    }
    GuassianArray.iterator
  }

  def REMOLD_compute_Inv_sqrt_det(var_matrix: DenseMatrix[Double], ratio: Double): (DenseMatrix[Double], Double, Int) = {
    ///SVD

    val svd.SVD(u, s, v) = svd(var_matrix)
    var r = 0
    var log_sqrt_det = 0.0
    val t_sum = s.data.sum
    val threshold = t_sum * ratio
    var t_tmp = 0.0
    while (r < s.data.length && t_tmp < threshold) {
      if (s.data(r) < numeric_lower_bound)
        s.data(r) = numeric_lower_bound
      log_sqrt_det += math.log(s.data(r))
      t_tmp += s.data(r)
      s.data(r) = 1.0/s.data(r)
      r += 1
    }

    val r_u = u(::, 0 until r)
    println("r = " + r)
    if (r < 1) {
      //println("Data size is " + size)
      println("Covariance is 0 !!!")
      //throw new Exception("Covariance is 0 !!!")
      s.data(0) = 1.0/numeric_lower_bound
      r = 1
    }

    return (r_u * diag(s(0 until r)) * r_u.t, 0.5*log_sqrt_det, r)
  }

  def CGM_compute_Mu_Inv_sqrt_det_sig(var_matrix: DenseMatrix[Double], svd_ratio: Double, size: Int): (DenseMatrix[Double], DenseVector[Double], Double, Double) = {
    ///SVD
    val svd.SVD(u, s, v) = svd(var_matrix)
    var r = 0
    var log_sqrt_det = 0.0
    val compression_ratio = svd_ratio
    val feature_value_threshold = s.data.sum * compression_ratio
    var feature_value_tmp = 0.0
    while (r < s.data.length && feature_value_tmp <= feature_value_threshold) {
      if (s.data(r) < numeric_lower_bound)
        s.data(r) = numeric_lower_bound
      feature_value_tmp += s.data(r)
      log_sqrt_det += math.log(s.data(r))
      s.data(r) = 1.0/s.data(r)
      r += 1
    }
    println("r = " + r)
    if (r < 1) {
      println("Data size is " + size)
      println("Covariance is 0 !!!")
      //throw new Exception("Covariance is 0 !!!")
      s.data(0) = 1.0/numeric_lower_bound
      r = 1
    }
    if (r == s.data.length) {
      r -= 1
    }
    val r_u = u(::, 0 until r)
    val sig = math.max(s(r until s.length).toArray.sum, numeric_lower_bound)/ (s.data.length - r)
    //P,S-1,sqrt_det,sig
    return (r_u, s(0 until r), 0.5*log_sqrt_det, sig)
  }

  def CGM_compute_PtMu_t_multiple_S_PtMu(a: DenseVector[Double], s: DenseVector[Double], b: DenseVector[Double]): Double = {
    val tmp = a :* s
    (tmp :* b).sum
  }

  def REMOLD_getGuassianModel(iterator: Iterator[GuassianData], dimension: Int, ratio: Double): Iterator[REMOLD_GuassianModel] = {
    val iter_arr = iterator.toArray
    val guassianModelCount = iter_arr.length
    val GuassianModelArray = new Array[REMOLD_GuassianModel](guassianModelCount)
    var i = 0
    for (model <- iter_arr) {
      val datamatrix = DenseMatrix.zeros[Double](model.data.length, dimension)
      for (r <- 0 until model.data.length){
        datamatrix(r,::) := model.data(r).t
      }
      GuassianModelArray(i) = new REMOLD_GuassianModel(dimension, model.partitionId, model.modelId)
      GuassianModelArray(i).mean = mean(datamatrix, Axis._0).t
      val tmp = datamatrix(*, ::) - GuassianModelArray(i).mean
      val N = datamatrix.rows.toDouble
      val varMatrix = (tmp.t * tmp) / N
      val inv_sqrt_det = REMOLD_compute_Inv_sqrt_det(varMatrix, ratio)
      GuassianModelArray(i).inverseMatrix = inv_sqrt_det._1
      GuassianModelArray(i).logSqrtDet = inv_sqrt_det._2
      GuassianModelArray(i).datasize = N
      GuassianModelArray(i).r = inv_sqrt_det._3
      i += 1
    }
    GuassianModelArray.iterator
  }

  def CGM_getGuassianModel(iterator: Iterator[GuassianData], dimension: Int, svd_ratio: Double): Iterator[CGM_GuassianModel] = {
    val iter_arr = iterator.toArray
    val guassianModelCount = iter_arr.length
    val GuassianModelArray = new Array[CGM_GuassianModel](guassianModelCount)
    var i = 0
    for (model <- iter_arr) {
      val datamatrix = DenseMatrix.zeros[Double](model.data.length, dimension)
      for (r <- 0 until model.data.length){
        datamatrix(r,::) := model.data(r).t
      }
      GuassianModelArray(i) = new CGM_GuassianModel(dimension, model.partitionId, model.modelId)
      val mu = mean(datamatrix, Axis._0).t
      val tmp = datamatrix(*, ::) - mu
      val N = datamatrix.rows.toDouble
      val varMatrix = (tmp.t * tmp) / N
      //mu,P,S-1,sqrt_det,sig
      val mu_P_PSP_sqrt_det_sig = CGM_compute_Mu_Inv_sqrt_det_sig(varMatrix, svd_ratio, tmp.rows)
      GuassianModelArray(i).mean = mu
      GuassianModelArray(i).P_Matrix = mu_P_PSP_sqrt_det_sig._1
      GuassianModelArray(i).dig_invMatrix = mu_P_PSP_sqrt_det_sig._2
      GuassianModelArray(i).logSqrtDet = mu_P_PSP_sqrt_det_sig._3
      GuassianModelArray(i).sig = mu_P_PSP_sqrt_det_sig._4
      GuassianModelArray(i).datasize = N
      i += 1
    }
    GuassianModelArray.iterator
  }

  def main(args: Array[String]): Unit = {

    val masterUrl = "spark://Master1:7077"
    val applicationName = "CGM_REMOLD"
    val conf = new SparkConf().setMaster(masterUrl).setAppName(applicationName).set("spark.kryoserializer.buffer.max", "2000m").set("spark.kryoserializer.buffer", "256m").set("spark.driver.maxResultSize", "55g").set("spark.shuffle.blockTransferService", "nio")//.set("spark.executor.memory", "10g")
    //SET Configration
    val sc = new SparkContext(conf)

    //        val datasetName = "50000_2_20_G-2.txt"
    //        val partitionNum = 4
    //        val dataset_size = 50000
    //        val dimension = 2
    //        val dataSetPath = "/mfl/dataset/" + datasetName
    //        val clusterSize = 20
    //        val ratio = 1.0
    //        val hash_seed = 0
    //        val svd_ratio = 0.9999

    //
    val datasetName = args(0)
    val partitionNum = args(1).toInt
    val dataset_size = args(2).toInt
    val dimension = args(3).toInt
    val dataSetPath = "/mfl/dataset/" + datasetName
    val clusterSize = args(4).toInt
    val ratio = args(5).toFloat
    val hash_seed = args(6).toInt
    val svd_ratio = args(7).toFloat
    val partitionK = math.sqrt(dataset_size/partitionNum).toInt
    //val partitionK = (math.sqrt(dataset_size).toInt).toInt

    println("dataset_size: "+dataset_size+" partitionK: "+partitionK)

    val size_per_partition = math.ceil(dataset_size.toDouble/partitionNum).toInt
    val numHashTables = 1
    val callableSize = 3
    val sqlContent = new SQLContext(sc)
    import sqlContent.implicits._

    var start_time = System.nanoTime()

//    val datasetOri = sc.textFile(dataSetPath, partitionNum).repartition(partitionNum).map(_.split(" ").map(_.toDouble)).map(x => (x.slice(0, dimension), x(dimension))).cache()
//    val rddGuassianData = datasetOri.mapPartitionsWithIndex((index, partition) => getGuassianData(index, partition, dimension, ratio, partitionK))
//    rddGuassianData.cache()




//    val datasetOri = sc.textFile(dataSetPath, partitionNum).repartition(partitionNum).map(_.split(" ").map(_.toDouble)).map(x => (x.slice(0, dimension), x(dimension)))
//    val index_data_label = datasetOri.zipWithIndex().map(x => (x._2, x._1._1, x._1._2))
//    val index_data = index_data_label.map(x => (x._1, Vectors.dense(x._2)))
//    val DF_index_data = index_data.toDF("index", "vec")
//
//
//    val brp = new BucketedRandomProjectionLSH().setBucketLength(1).setNumHashTables(numHashTables).setInputCol("vec").setOutputCol("lshvalues").setSeed(hash_seed)
//
//
//    val model = brp.fit(DF_index_data)
//    val transform_DF_index_data = model.transform(DF_index_data).cache()
//
//
//    val collect_index_hash = transform_DF_index_data.rdd.map(x => x.getValuesMap[Double](Array("index", "lshvalues"))).map(x => (x.getOrElse("index", -1).asInstanceOf[Long], x.getOrElse("lshvalues", doubleWrapper(-1.0)).asInstanceOf[WrappedArray[Double]].mkString(",").split(',').map(t => t.substring(1, t.length - 1).toDouble).sum)).collect()
//    val index_to_partitionID = new scala.collection.mutable.HashMap[Long, Int]
//    val index_hash_sorted = collect_index_hash.sortBy(_._2)
//    var count_t = 0
//    var partitionID = 0
//    for (i <- 0 until index_hash_sorted.length){
//      index_to_partitionID(index_hash_sorted(i)._1) = partitionID
//      count_t += 1
//      if (count_t == size_per_partition){
//        partitionID += 1
//        count_t = 0
//      }
//    }
//    index_data_label.count()
//
//    val tmp = index_data_label.map(x => (index_to_partitionID(x._1), (x._2, x._3))).partitionBy(new LSHPartitioner(partitionNum)).map(x => x._2)
//    tmp.count()
//    val LSH_dataset_label = tmp.map(x => (new DenseVector[Double](x._1), x._2))
//    LSH_dataset_label.count()

    val datasetOri = sc.textFile(dataSetPath, partitionNum).repartition(partitionNum).map(_.split(" ").map(_.toDouble)).cache()
    val dfdataset = datasetOri.map(x => (Vectors.dense(x.slice(0, dimension)), x(dimension))).toDF("vec", "label").cache()

    val brp = new BucketedRandomProjectionLSH().setBucketLength(1).setNumHashTables(numHashTables).setInputCol("vec").setOutputCol("lshvalues").setSeed(hash_seed)
    val model = brp.fit(dfdataset)
    val transform_dataset = model.transform(dfdataset).cache()
    val dataset_LSH_label = transform_dataset.rdd.map(x => x.getValuesMap[Double](List("vec", "label", "lshvalues"))).map(x => (x.getOrElse("lshvalues", doubleWrapper(-1.0)).asInstanceOf[WrappedArray[Double]].mkString(",").split(',').map(t => t.substring(1, t.length - 1).toDouble).sum, removefirst_orlast(x.get("vec").toVector.mkString(",")).split(',').map(_.toDouble), x.getOrElse("label", -1).asInstanceOf[Double])).cache()
    val dataset_label = dataset_LSH_label.sortBy(x => x._1).zipWithIndex.map { case (k, v) => (v, (k._2, k._3)) }.cache()
    val LSH_dataset_label = dataset_label.partitionBy(new LSHPartitioner(partitionNum, ceil(dataset_size * 1.0 / partitionNum).toInt)).map(x => (new DenseVector[Double](x._2._1), x._2._2)).cache()

    val rddGuassianData = LSH_dataset_label.mapPartitionsWithIndex((index, partition) => getGuassianData(index, partition, dimension, ratio, partitionK))
    rddGuassianData.cache()
    val local_information = rddGuassianData.map(x => (x.partitionId, x.modelId , x.data.size, x.label)).collect()




    //local_label.collect()
    //val local_data = rddGuassianData.collect()
    var end_time = System.nanoTime()
    val stage0_time = end_time-start_time
    println("Stage 0 time: "+(stage0_time/1e9)+"s")
    //////第一阶段结束
    val threadPool: ExecutorService = Executors.newFixedThreadPool(callableSize)
    val diff = 1e-6

    val one_three = 1.0 / 3.0
    val two_three = 2.0 / 3.0

    val alpha_lower_bound = -1e50
    var count_bytes = 0.0


    //////CGM 阶段开始

    println("\nCGM:")
    start_time = System.nanoTime()
    val CGM_startTime_total = System.nanoTime()
    val CGM_start_time = System.nanoTime()
    var CGM_best_str_nmi = ""
    var CGM_best_str_purity = ""
    var CGM_best_nmi = -1.0
    var CGM_bestNmi_purity = -1.0
    var CGM_nmi_E_array = new ArrayBuffer[Double]()
    var CGM_nmi_E2_array = new ArrayBuffer[Double]()
    var CGM_best_purity = -1.0
    var CGM_purity_E_array = new ArrayBuffer[Double]()
    var CGM_purity_E2_array = new ArrayBuffer[Double]()
    var CGM_best_alpha_nmi = 0.0
    var CGM_best_alpha_purity = 0.0
    var CGM_best_cluster_res_nmi: Array[String] = null
    var CGM_best_cluster_res_purity: Array[String] = null

    val CGM_rddGuassianModel = rddGuassianData.mapPartitions(partition => CGM_getGuassianModel(partition, dimension, svd_ratio)).cache()
    CGM_rddGuassianModel.count()
    println("CGM_rddGuassianModel.collect() begin.")
    var CGM_localGuassianModel = CGM_rddGuassianModel.collect()
    val CGM_guassianModelNum = CGM_localGuassianModel.length
    println("CGM_rddGuassianModel.collect() succeed. #Modles: "+CGM_guassianModelNum)
    CGM_rddGuassianModel.unpersist()
    rddGuassianData.unpersist()
    LSH_dataset_label.unpersist()
    dataset_label.unpersist()
    dataset_LSH_label.unpersist()
    datasetOri.unpersist()
    dfdataset.unpersist()


    //    val CGM_t = (sqrt(CGM_guassianModelNum) * ratio).toInt
    val CGM_t = math.max((sqrt(CGM_guassianModelNum) * ratio).toInt+1, 8)
    val CGM_guassianModelK = if (CGM_t > CGM_guassianModelNum) CGM_guassianModelNum else CGM_t

    val CGM_local_means = CGM_localGuassianModel.map(_.mean).par

    val CGM_index_KNN = new Array[Array[Int]](CGM_guassianModelNum)
    val CGM_listTmp = new Array[(Int, Double)](CGM_guassianModelNum)

    for (i <- 0 until CGM_guassianModelNum) {
      val x_arr = CGM_local_means(i)
      for (j <- 0 until CGM_guassianModelNum) {
        val y_arr = CGM_local_means(j)
        CGM_listTmp(j) = Tuple2(j, squaredEuclideanDis(x_arr, y_arr))
      }
      buildHeap(CGM_listTmp)
      CGM_index_KNN(i) = topK(CGM_listTmp, CGM_guassianModelK).map(_._1)
    }

    val P_Matrix = new Array[Array[Double]](CGM_guassianModelNum)


    val PtMu = new Array[Array[DenseVector[Double]]](CGM_guassianModelNum)
    val Mu_diff_PPtMu = new Array[Array[DenseVector[Double]]](CGM_guassianModelNum)

    for (i <- 0 until CGM_guassianModelNum) {
      PtMu(i) = new Array[DenseVector[Double]](CGM_guassianModelNum)
      Mu_diff_PPtMu(i) = new Array[DenseVector[Double]](CGM_guassianModelNum)
      for (j <- 0 until CGM_guassianModelNum) {
        PtMu(i)(j) = CGM_localGuassianModel(i).P_Matrix.t * CGM_localGuassianModel(j).mean
        Mu_diff_PPtMu(i)(j) = CGM_localGuassianModel(j).mean - CGM_localGuassianModel(i).P_Matrix * PtMu(i)(j)
      }
    }

    val CGM_alphaMatrix: DenseMatrix[Double] = DenseMatrix.ones[Double](CGM_guassianModelNum, CGM_guassianModelNum)*alpha_lower_bound
    val CGM_visMatrix: DenseMatrix[Boolean] = DenseMatrix.eye[Boolean](CGM_guassianModelNum)

    var mean_of_log_pow_sig = 0.0
    var mean_of_logSqrtDet = 0.0
    for (i <- 0 until CGM_guassianModelNum){
      mean_of_log_pow_sig += ((CGM_localGuassianModel(i).P_Matrix.rows - CGM_localGuassianModel(i).P_Matrix.cols) / 2.0) *log(CGM_localGuassianModel(i).sig)
      mean_of_logSqrtDet += CGM_localGuassianModel(i).logSqrtDet
    }
    mean_of_log_pow_sig /= CGM_guassianModelNum
    mean_of_logSqrtDet /= CGM_guassianModelNum


    var CGM_max_alpha_all = alpha_lower_bound
    var CGM_min_alpha_all = 1.0
    for (i <- 0 until CGM_guassianModelNum) {
      for (j <- CGM_index_KNN(i)) {
        if (CGM_visMatrix(i, j) == false){
          val arrfuture = new Array[FutureTask[Double]](callableSize)
          val arrmin = new Array[Double](callableSize)
          val union_knnset = (CGM_index_KNN(i) ++ CGM_index_KNN(j)).distinct

          val XY_pri_2_map = new mutable.HashMap[Int, DenseMatrix[Double]]
          for (index <- union_knnset) {
            XY_pri_2_map(index) = DenseMatrix.zeros(3, 3)
            XY_pri_2_map(index)(0, 0) = Mu_diff_PPtMu(index)(i).t * Mu_diff_PPtMu(index)(i)
            XY_pri_2_map(index)(0, 1) = Mu_diff_PPtMu(index)(i).t * Mu_diff_PPtMu(index)(j)
            XY_pri_2_map(index)(0, 2) = Mu_diff_PPtMu(index)(i).t * Mu_diff_PPtMu(index)(index)
            XY_pri_2_map(index)(1, 1) = Mu_diff_PPtMu(index)(j).t * Mu_diff_PPtMu(index)(j)
            XY_pri_2_map(index)(1, 2) = Mu_diff_PPtMu(index)(j).t * Mu_diff_PPtMu(index)(index)
            XY_pri_2_map(index)(2, 2) = Mu_diff_PPtMu(index)(index).t * Mu_diff_PPtMu(index)(index)
          }

          val XY_pri_1_map = new mutable.HashMap[Int, DenseMatrix[Double]]

          for (index <- union_knnset) {
            XY_pri_1_map(index) = DenseMatrix.zeros(3, 3)
            XY_pri_1_map(index)(0, 0) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(i), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(i))
            XY_pri_1_map(index)(0, 1) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(i), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(j))
            XY_pri_1_map(index)(0, 2) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(i), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(index))
            XY_pri_1_map(index)(1, 1) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(j), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(j))
            XY_pri_1_map(index)(1, 2) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(j), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(index))
            XY_pri_1_map(index)(2, 2) = CGM_compute_PtMu_t_multiple_S_PtMu(PtMu(index)(index), CGM_localGuassianModel(index).dig_invMatrix, PtMu(index)(index))
          }

          for (k <- 0 until callableSize) {
            arrfuture(k) = new FutureTask[Double](new Callable[Double] {
              override def call(): Double = {
                var alpha_left = 1.0 * k / callableSize
                var alpha_right = 1.0 * (k + 1) / callableSize

                var min_value = min(CGM_computeAllModel(mean_of_log_pow_sig, mean_of_logSqrtDet, alpha_left, i, j, union_knnset, CGM_localGuassianModel, XY_pri_1_map, XY_pri_2_map), CGM_computeAllModel(mean_of_log_pow_sig, mean_of_logSqrtDet, alpha_right, i, j, union_knnset, CGM_localGuassianModel, XY_pri_1_map, XY_pri_2_map))
                var alpha1 = alpha_left
                var alpha2 = alpha_right
                while ((alpha2 - alpha1) > diff) {
                  alpha1 = (alpha_right - alpha_left) * one_three + alpha_left
                  alpha2 = (alpha_right - alpha_left) * two_three + alpha_left
                  val leftValue = CGM_computeAllModel(mean_of_log_pow_sig, mean_of_logSqrtDet, alpha1, i, j, union_knnset, CGM_localGuassianModel, XY_pri_1_map, XY_pri_2_map)
                  val rightValue = CGM_computeAllModel(mean_of_log_pow_sig, mean_of_logSqrtDet, alpha2, i, j, union_knnset, CGM_localGuassianModel, XY_pri_1_map, XY_pri_2_map)
                  if (leftValue > rightValue) {
                    alpha_left = alpha1
                    min_value = min(rightValue, min_value)
                  }
                  else {
                    alpha_right = alpha2
                    min_value = min(leftValue, min_value)
                  }
                }
                return min_value
              }
            })
            threadPool.execute(arrfuture(k))
          }
          for (k <- 0 until callableSize) {
            arrmin(k) = arrfuture(k).get()
          }
          val min_all_value = arrmin.min
          CGM_alphaMatrix(i, j) = min_all_value
          CGM_alphaMatrix(j, i) = CGM_alphaMatrix(i, j)
          CGM_max_alpha_all = math.max(CGM_max_alpha_all, min_all_value)
          CGM_min_alpha_all = math.min(CGM_min_alpha_all, min_all_value)
          CGM_visMatrix(i, j) = true
          CGM_visMatrix(j, i) = true
        }
      }
    }

    end_time = System.nanoTime()
    println("CGM stage 1 time: " + ((end_time-start_time)/1e9)+"s")
    println(s"CGM_alphaMatrix max: $CGM_max_alpha_all  CGM_alphaMatrix min:  $CGM_min_alpha_all")
    //    val CGM_range = max(CGM_alphaMatrix)-min(CGM_alphaMatrix)
    //    CGM_alphaMatrix -= min(CGM_alphaMatrix)
    //    CGM_alphaMatrix /= CGM_range

    val CGM_alpha_res = new ArrayBuffer[Double]()
    var CGM_all_alpha_arr = new ArrayBuffer[(Double, Int, Int)]()


    for (i <- 0 until CGM_alphaMatrix.rows) {
      for (j <- i+1 until CGM_alphaMatrix.cols) {
        CGM_all_alpha_arr += Tuple3(CGM_alphaMatrix(i, j), i, j)
      }
    }

    CGM_all_alpha_arr = CGM_all_alpha_arr.sortBy(x => -x._1)
    val CGM_top_k = CGM_localGuassianModel.map(x => x.dig_invMatrix.length).sum * 1.0 / CGM_localGuassianModel.length
    var CGM_all_nmi_res = new ArrayBuffer[Double]()
    var CGM_all_purity_res = new ArrayBuffer[Double]()
    //var CGM_all_alpha_res = new ArrayBuffer[Double]()

    val CGM_local_id_to_global_id = new Array[mutable.HashMap[Int, Int]](rddGuassianData.partitions.size)
    for (i <- 0 until rddGuassianData.partitions.size) {
      CGM_local_id_to_global_id(i) = new mutable.HashMap[Int, Int]
    }
    for (i <- 0 until CGM_guassianModelNum) {
      CGM_local_id_to_global_id(CGM_localGuassianModel(i).partitionId)(CGM_localGuassianModel(i).modelId) = i
    }

    //val CGM_writer_model = new PrintWriter(new File("model/CGM_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_model.txt"))

    for (model <- CGM_localGuassianModel) {
      count_bytes += model.P_Matrix.rows*(model.P_Matrix.cols+1)+model.P_Matrix.cols+1
      //CGM_writer_model.write(model.partitionId.toString + '\t' + model.modelId.toString + '\t' + model.datasize.toString + '\t' + model.mean.toArray.mkString(",") + '\t' + model.P_Matrix.toArray.mkString(",") + '\t' + model.logSqrtDet.toString + '\t' + model.dig_invMatrix.toArray.mkString(",") + '\t' + model.sig + "\n")
    }
    count_bytes *= 4
    count_bytes += dataset_size*2
    println("count_bytes: "+(count_bytes/(1024*1024))+" MB")

    CGM_localGuassianModel = null
    //CGM_localGuassianModel.re
    //CGM_writer_model.close()

    val CGM_end_time = System.nanoTime() //获取结束时间


    //count_double += 4*local_information.length
    val CGM_clusterUnionFindSet_alpha = new UnionFindSet(CGM_guassianModelNum)
    for (tuple3 <- CGM_all_alpha_arr) {
      val alpha = tuple3._1
      val i = tuple3._2
      val j = tuple3._3
      if (CGM_clusterUnionFindSet_alpha.find(i) != CGM_clusterUnionFindSet_alpha.find(j)) {

        CGM_clusterUnionFindSet_alpha.union(i, j)
        val arrLabels = new Array[LabelPredict](local_information.size)
        for (i <- 0 until local_information.size) {
          val information = local_information(i)
          arrLabels(i) = new LabelPredict(information._3, information._4.toArray, CGM_clusterUnionFindSet_alpha.find(CGM_local_id_to_global_id(information._1)(information._2)))
          // arrLabels(i) = new DataWithLabelPredict(partition_data.label.toArray, arrAutonicMap(partition_data.partitionId)(partition_data.modelId))
        }


        val res_local = arrLabels.flatMap(x => x.LabelPredict)
        val label1 = res_local.map(x => x(1))
        val label2 = res_local.map(x => x(0))


        val CGM_n = res_local.length
        val CGM_nmi_and_predict_cluster_size = cal_NMI(label1, label2, CGM_n)
        val CGM_nmi = CGM_nmi_and_predict_cluster_size._1
        val CGM_predict_cluster_size_nmi = CGM_nmi_and_predict_cluster_size._2

        val CGM_purity_and_predict_cluster_size = cal_purity(label1, label2, CGM_n)
        val CGM_purity = CGM_purity_and_predict_cluster_size._1
        val CGM_predict_cluster_size_purity = CGM_purity_and_predict_cluster_size._2
        CGM_nmi_E_array += CGM_nmi
        CGM_purity_E_array += CGM_purity
        CGM_nmi_E2_array += pow(CGM_nmi, 2)
        CGM_purity_E2_array += pow(CGM_purity, 2)

        CGM_all_nmi_res += CGM_nmi
        CGM_all_purity_res += CGM_purity
        CGM_alpha_res += alpha

        if (CGM_nmi > CGM_best_nmi) {
          val t_str_nmi = "alpha = " + alpha + " hash_value = " + hash_seed + " nmi = " + CGM_nmi + " purity = " + CGM_purity + " predict_cluster_size = " + CGM_predict_cluster_size_nmi + " svd_top_k = " + CGM_top_k + " Model 个数 " + CGM_guassianModelNum + " Running time " + (stage0_time + CGM_end_time - CGM_start_time) / 1e9 + "s"
          CGM_best_nmi = CGM_nmi
          CGM_bestNmi_purity = CGM_purity
          CGM_best_str_nmi = t_str_nmi
          CGM_best_alpha_nmi = alpha
          //CGM_best_cluster_res_nmi = res_local.map(x => x.mkString(","))
        }
        if (CGM_purity > CGM_best_purity) {
          val t_str_purity = "alpha = " + alpha + " hash_value = " + hash_seed + " purity = " + CGM_purity + " predict_cluster_size = " + CGM_predict_cluster_size_purity + " svd_top_k = " + CGM_top_k + " Model 个数 " + CGM_guassianModelNum + " Running time " + (stage0_time + CGM_end_time - CGM_start_time) / 1e9 + "s"
          CGM_best_purity = CGM_purity
          CGM_best_str_purity = t_str_purity
          CGM_best_alpha_purity = alpha
          //CGM_best_cluster_res_purity = res_local.map(x => x.mkString(","))
        }
      }
    }

    val CGM_all_nmi_res_sorted = CGM_all_nmi_res.sortBy(x => -x)
    CGM_all_nmi_res_sorted.trimEnd((CGM_all_nmi_res_sorted.length * 0.8).toInt)
    val CGM_all_purity_res_sorted = CGM_all_purity_res.sortBy(x => -x)
    CGM_all_purity_res_sorted.trimEnd((CGM_all_purity_res_sorted.length * 0.8).toInt)

    val CGM_endTime_total = System.nanoTime()
    println("Running_time_total=" + (stage0_time+CGM_endTime_total - CGM_startTime_total) / 1e9 + "s")
    println(CGM_best_str_nmi)
    val CGM_nmi_mean = CGM_nmi_E_array.sum / CGM_alpha_res.length
    val CGM_nmi_var = CGM_nmi_E2_array.sum / CGM_alpha_res.length - pow(CGM_nmi_mean, 2)
    println("nmi mean = " + CGM_nmi_mean + "  nmi var = " + CGM_nmi_var)
    println("nmi mean top 20% = " + CGM_all_nmi_res_sorted.sum / CGM_all_nmi_res_sorted.length + "   nmi var top 20% = " + (CGM_all_nmi_res_sorted.map(x => x * x).sum / CGM_all_nmi_res_sorted.length - pow(CGM_all_nmi_res_sorted.sum / CGM_all_nmi_res_sorted.length, 2)))

    println(CGM_best_str_purity)
    val CGM_purity_mean = CGM_purity_E_array.sum / CGM_alpha_res.length
    val CGM_purity_var = CGM_purity_E2_array.sum / CGM_alpha_res.length - pow(CGM_purity_mean, 2)
    println("purity mean = " + CGM_purity_mean + "  purity var = " + CGM_purity_var)
    println("purity mean top 20% = " + CGM_all_purity_res_sorted.sum / CGM_all_purity_res_sorted.length + "   purity var top 20% = " + (CGM_all_purity_res_sorted.map(x => x * x).sum / CGM_all_purity_res_sorted.length - pow(CGM_all_purity_res_sorted.sum / CGM_all_purity_res_sorted.length, 2)))
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("number of alphas: "+CGM_all_nmi_res.length)
    println("nmi:")
    for (nmi_each <- CGM_all_nmi_res){
      print(nmi_each.formatted("%.3f")+" ")
    }
    println()
    println("purity:")
    for (purity_each <- CGM_all_purity_res){
      print(purity_each.formatted("%.3f")+" ")
    }
    println()
    println("alpha:")
    for (alpha_each <- CGM_alpha_res){
      print(alpha_each+" ")
    }
    println()



//    val CGM_writer_cluster_res_nmi = new PrintWriter(new File("cluster_res/CGM_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_cluster_res_nmi.txt"))
//    for (res_line <- CGM_best_cluster_res_nmi) {
//      CGM_writer_cluster_res_nmi.write(res_line + "\n")
//    }
//    CGM_writer_cluster_res_nmi.close()
//
//    val CGM_writer_cluster_res_purity = new PrintWriter(new File("cluster_res/CGM_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_cluster_res_purity.txt"))
//    for (res_line <- CGM_best_cluster_res_purity) {
//      CGM_writer_cluster_res_purity.write(res_line + "\n")
//    }
//    CGM_writer_cluster_res_purity.close()



    ////REMOLD 阶段开始
//    println("\nREMOLD:")
//    start_time = System.nanoTime()
//    val REMOLD_startTime_total = System.nanoTime()
//    val REMOLD_start_time = System.nanoTime()
//    var REMOLD_best_str_nmi = ""
//    var REMOLD_best_str_purity = ""
//    var REMOLD_best_nmi = -1.0
//    var REMOLD_bestNmi_purity = -1.0
//    var REMOLD_nmi_E_array = new ArrayBuffer[Double]()
//    var REMOLD_nmi_E2_array = new ArrayBuffer[Double]()
//    var REMOLD_best_purity = -1.0
//    var REMOLD_purity_E_array = new ArrayBuffer[Double]()
//    var REMOLD_purity_E2_array = new ArrayBuffer[Double]()
//    var REMOLD_best_alpha_nmi = 0.0
//    var REMOLD_best_alpha_purity = 0.0
//    var REMOLD_best_cluster_res_nmi: Array[String] = null
//    var REMOLD_best_cluster_res_purity: Array[String] = null
//
//    val REMOLD_rddGuassianModel = rddGuassianData.mapPartitions(partition => REMOLD_getGuassianModel(partition, dimension, svd_ratio))
//
//    println("REMOLD_rddGuassianModel.collect() begin.")
//    var REMOLD_localGuassianModel = REMOLD_rddGuassianModel.collect()
//    val REMOLD_guassianModelNum = REMOLD_localGuassianModel.length
//    println("REMOLD_localGuassianModel.collect() succeed. #Modles: "+REMOLD_guassianModelNum)
//
//    REMOLD_rddGuassianModel.unpersist()
//
//    val REMOLD_t = math.max((sqrt(REMOLD_guassianModelNum) * ratio).toInt+1, 8)
//    val REMOLD_guassianModelK = if (REMOLD_t > REMOLD_guassianModelNum) REMOLD_guassianModelNum else REMOLD_t
//
//    val REMOLD_local_means = REMOLD_localGuassianModel.map(_.mean).par
//
//    val REMOLD_index_KNN = new Array[Array[Int]](REMOLD_guassianModelNum)
//    val REMOLD_listTmp = new Array[(Int, Double)](REMOLD_guassianModelNum-1)
//
//    for (i <- 0 until REMOLD_guassianModelNum) {
//      val x_arr = REMOLD_local_means(i)
//      var tmp_i = 0
//      for (j <- 0 until REMOLD_guassianModelNum)
//        if(i != j){
//          val y_arr = REMOLD_local_means(j)
//          REMOLD_listTmp(tmp_i) = Tuple2(j, squaredEuclideanDis(x_arr, y_arr))
//          tmp_i += 1
//        }
//      buildHeap(REMOLD_listTmp)
//      REMOLD_index_KNN(i) = topK(REMOLD_listTmp, REMOLD_guassianModelK).map(_._1)
//    }
//
//    val XYX_Matrix = new Array[Array[Double]](REMOLD_guassianModelNum)
//    val XYY_Matrix = new Array[Array[Double]](REMOLD_guassianModelNum)
//
//    for (i <- 0 until REMOLD_guassianModelNum) {
//      val x_arr = REMOLD_localGuassianModel(i).mean
//      XYX_Matrix(i) = new Array[Double](REMOLD_guassianModelNum)
//      XYY_Matrix(i) = new Array[Double](REMOLD_guassianModelNum)
//      for (j <- 0 until REMOLD_guassianModelNum) {
//        val y_arr = REMOLD_localGuassianModel(j).mean
//        val y_matrix = REMOLD_localGuassianModel(j).inverseMatrix
//        XYX_Matrix(i)(j) = x_arr.t * y_matrix * x_arr
//        XYY_Matrix(i)(j) = x_arr.t * y_matrix * y_arr
//      }
//    }
//
//    //val REMOLD_alphaMatrix: DenseMatrix[Double] = DenseMatrix.eye[Double](REMOLD_guassianModelNum)
//
//    val REMOLD_alphaMatrix: DenseMatrix[Double] = DenseMatrix.ones[Double](REMOLD_guassianModelNum, REMOLD_guassianModelNum)*alpha_lower_bound
//    val REMOLD_visMatrix: DenseMatrix[Boolean] = DenseMatrix.eye[Boolean](REMOLD_guassianModelNum)
//
//    var REMOLD_max_alpha_all = alpha_lower_bound
//    var REMOLD_min_alpha_all = 1.0
//
//
//    for (i <- 0 until REMOLD_guassianModelNum) {
//      for (j <- REMOLD_index_KNN(i)) {
//        if (REMOLD_visMatrix(i, j) == false){
//          val arrfuture = new Array[FutureTask[Double]](callableSize)
//          val arrmin = new Array[Double](callableSize)
//          val union_knnset = (REMOLD_index_KNN(i) ++ REMOLD_index_KNN(j)).distinct
//
//          val XYZ_arr = new mutable.HashMap[Int, Double]
//          for (index <- union_knnset) {
//            XYZ_arr(index) = REMOLD_localGuassianModel(i).mean.t * REMOLD_localGuassianModel(index).inverseMatrix * REMOLD_localGuassianModel(j).mean
//          }
//
//          for (k <- 0 until callableSize) {
//            arrfuture(k) = new FutureTask[Double](new Callable[Double] {
//              override def call(): Double = {
//                var alpha_left = 1.0 * k / callableSize
//                var alpha_right = 1.0 * (k + 1) / callableSize
//                var min_value = min(REMOLD_computeAllModel(alpha_left, i, j, union_knnset, REMOLD_localGuassianModel, XYX_Matrix, XYY_Matrix, XYZ_arr, dimension), REMOLD_computeAllModel(alpha_right, i, j, union_knnset, REMOLD_localGuassianModel, XYX_Matrix, XYY_Matrix, XYZ_arr, dimension))
//                var alpha1 = alpha_left
//                var alpha2 = alpha_right
//
//                while ((alpha2 - alpha1) > diff) {
//                  alpha1 = (alpha_right - alpha_left) * one_three + alpha_left
//                  alpha2 = (alpha_right - alpha_left) * two_three + alpha_left
//                  val leftValue = REMOLD_computeAllModel(alpha1, i, j, union_knnset, REMOLD_localGuassianModel, XYX_Matrix, XYY_Matrix, XYZ_arr, dimension)
//                  val rightValue = REMOLD_computeAllModel(alpha2, i, j, union_knnset, REMOLD_localGuassianModel, XYX_Matrix, XYY_Matrix, XYZ_arr, dimension)
//                  if (leftValue > rightValue) {
//                    alpha_left = alpha1
//                    min_value = min(rightValue, min_value)
//                  }
//                  else {
//                    alpha_right = alpha2
//                    min_value = min(leftValue, min_value)
//                  }
//                }
//                return min_value
//              }
//            })
//            threadPool.execute(arrfuture(k))
//          }
//          for (k <- 0 until callableSize) {
//            arrmin(k) = arrfuture(k).get()
//          }
//          val min_all_value = arrmin.min
//          REMOLD_alphaMatrix(i, j) = min_all_value
//          REMOLD_alphaMatrix(j, i) = REMOLD_alphaMatrix(i, j)
//          REMOLD_max_alpha_all = math.max(REMOLD_max_alpha_all, min_all_value)
//          REMOLD_min_alpha_all = math.min(REMOLD_min_alpha_all, min_all_value)
//          REMOLD_visMatrix(i, j) = true
//          REMOLD_visMatrix(j, i) = true
//        }
//      }
//    }
//
////    println("REMOLD_alphaMatrix max: " + max(REMOLD_alphaMatrix))
//    end_time = System.nanoTime()
//    println("REMOLD stage 1 time: " + ((end_time-start_time)/1e9)+"s")
//    println(s"REMOLD_alphaMatrix max: $REMOLD_max_alpha_all  REMOLD_alphaMatrix min:  $REMOLD_min_alpha_all")
////    val REMOLD_range = max(REMOLD_alphaMatrix)-min(REMOLD_alphaMatrix)
////    REMOLD_alphaMatrix -= min(REMOLD_alphaMatrix)
////    REMOLD_alphaMatrix /= REMOLD_range
//
//    //REMOLD_alphaMatrix /= max(REMOLD_alphaMatrix)
//
//    var REMOLD_alpha_res = new ArrayBuffer[Double]()
//    val REMOLD_clusterUnionFindSet_alpha = new UnionFindSet(REMOLD_guassianModelNum)
//    var REMOLD_all_alpha_arr = new ArrayBuffer[(Double, Int, Int)]()
//
//    for (i <- 0 until REMOLD_alphaMatrix.rows) {
//      for (j <- i+1 until REMOLD_alphaMatrix.cols) {
//        REMOLD_all_alpha_arr += Tuple3(REMOLD_alphaMatrix(i, j), i, j)
//      }
//    }
//
//    REMOLD_all_alpha_arr = REMOLD_all_alpha_arr.sortBy(x => -x._1)
////    REMOLD_alpha_res += 1.0
//
//    val REMOLD_top_k = REMOLD_localGuassianModel.map(x => x.r).sum * 1.0 / REMOLD_localGuassianModel.length
//    var REMOLD_all_nmi_res = new ArrayBuffer[Double]()
//    var REMOLD_all_purity_res = new ArrayBuffer[Double]()
//
//    val REMOLD_local_id_to_global_id = new Array[mutable.HashMap[Int, Int]](rddGuassianData.partitions.size)
//    for (i <- 0 until rddGuassianData.partitions.size) {
//      REMOLD_local_id_to_global_id(i) = new mutable.HashMap[Int, Int]
//    }
//    for (i <- 0 until REMOLD_guassianModelNum) {
//      REMOLD_local_id_to_global_id(REMOLD_localGuassianModel(i).partitionId)(REMOLD_localGuassianModel(i).modelId) = i
//    }
//    //val bc_REMOLD_local_id_to_global = sc.broadcast(REMOLD_local_id_to_global_id)
//
//    count_bytes = 0.0
//    for (model <- REMOLD_localGuassianModel){
//      count_bytes += model.inverseMatrix.rows*(model.inverseMatrix.rows+1)
//    }
//    count_bytes *=4
//    count_bytes += dataset_size*partitionNum*2
//    println("count_bytes: "+(count_bytes/(1024*1024))+" MB")
//
//    REMOLD_localGuassianModel = null
//
//
//    val REMOLD_end_time = System.nanoTime() //获取结束时间
//
//    for (tuple3 <- REMOLD_all_alpha_arr) {
//      val alpha = tuple3._1
//      val i = tuple3._2
//      val j = tuple3._3
//      if (REMOLD_clusterUnionFindSet_alpha.find(i) != REMOLD_clusterUnionFindSet_alpha.find(j)) {
//        REMOLD_alpha_res += alpha
//        REMOLD_clusterUnionFindSet_alpha.union(i, j)
//
//        val arrLabels = new Array[LabelPredict](local_information.size)
//
//
//        for (i <- 0 until local_information.size) {
//          val information = local_information(i)
//          arrLabels(i) = new LabelPredict(information._3, information._4.toArray, REMOLD_clusterUnionFindSet_alpha.find(REMOLD_local_id_to_global_id(information._1)(information._2)))
//          // arrLabels(i) = new DataWithLabelPredict(partition_data.label.toArray, arrAutonicMap(partition_data.partitionId)(partition_data.modelId))
//        }
//
//        val res_local = arrLabels.flatMap(x => x.LabelPredict)
//
//        // label1 = predict  label2 = label
//        val label1 = res_local.map(x => x(1).toInt)
//        val label2 = res_local.map(x => x(0).toInt)
//        val n = res_local.length
//        val nmi_and_predict_cluster_size = cal_NMI(label1, label2, n)
//        val nmi = nmi_and_predict_cluster_size._1
//        val predict_cluster_size_nmi = nmi_and_predict_cluster_size._2
//
//        val purity_and_predict_cluster_size = cal_purity(label1, label2, n)
//        val purity = purity_and_predict_cluster_size._1
//        val predict_cluster_size_purity = purity_and_predict_cluster_size._2
//        REMOLD_nmi_E_array += nmi
//        REMOLD_purity_E_array += purity
//        REMOLD_nmi_E2_array += pow(nmi, 2)
//        REMOLD_purity_E2_array += pow(purity, 2)
//
//        REMOLD_all_nmi_res += nmi
//        REMOLD_all_purity_res += purity
//
//        if (nmi > REMOLD_best_nmi) {
//          val t_str_nmi = "alpha = " + alpha + " hash_value = " + hash_seed + " nmi = " + nmi + " purity = " + purity + " predict_cluster_size = " + predict_cluster_size_nmi + " svd_top_k = " + REMOLD_top_k + " Model 个数 " + REMOLD_guassianModelNum + " 运行时间为 " + (stage0_time + REMOLD_end_time - REMOLD_start_time) / 1e9 + "s"
//          REMOLD_best_nmi = nmi
//          REMOLD_bestNmi_purity = purity
//          REMOLD_best_str_nmi = t_str_nmi
//          REMOLD_best_alpha_nmi = alpha
//        //  REMOLD_best_cluster_res_nmi = res_local.map(x => x.mkString(","))
//        }
//        if (purity > REMOLD_best_purity) {
//          val t_str_purity = "alpha = " + alpha + " hash_value = " + hash_seed + " purity = " + purity + " predict_cluster_size = " + predict_cluster_size_purity + " svd_top_k = " + REMOLD_top_k + " Model 个数 " + REMOLD_guassianModelNum + " 运行时间为 " + (stage0_time + REMOLD_end_time - REMOLD_start_time) / 1e9 + "s"
//          REMOLD_best_purity = purity
//          REMOLD_best_str_purity = t_str_purity
//          REMOLD_best_alpha_purity = alpha
//         // REMOLD_best_cluster_res_purity = res_local.map(x => x.mkString(","))
//        }
//      }
//    }
//
//
//    REMOLD_all_nmi_res = REMOLD_all_nmi_res.sortBy(x => -x)
//    REMOLD_all_nmi_res.trimEnd((REMOLD_all_nmi_res.length * 0.8).toInt)
//    REMOLD_all_purity_res = REMOLD_all_purity_res.sortBy(x => -x)
//    REMOLD_all_purity_res.trimEnd((REMOLD_all_purity_res.length * 0.8).toInt)
//
//    val REMOLD_endTime_total = System.nanoTime()
//    println("Running_time_total=" + (stage0_time+REMOLD_endTime_total - REMOLD_startTime_total) / 1e9 + "s")
//    println(REMOLD_best_str_nmi)
//
//    val REMOLD_nmi_mean = REMOLD_nmi_E_array.sum / REMOLD_alpha_res.length
//    val REMOLD_nmi_var = REMOLD_nmi_E2_array.sum / REMOLD_alpha_res.length - pow(REMOLD_nmi_mean, 2)
//    println("nmi mean = " + REMOLD_nmi_mean + "  nmi var = " + REMOLD_nmi_var)
//    println("nmi mean top 20% = " + REMOLD_all_nmi_res.sum / REMOLD_all_nmi_res.length + "   nmi var top 20% = " + (REMOLD_all_nmi_res.map(x => x * x).sum / REMOLD_all_nmi_res.length - pow(REMOLD_all_nmi_res.sum / REMOLD_all_nmi_res.length, 2)))
//
//    println(REMOLD_best_str_purity)
//    val REMOLD_purity_mean = REMOLD_purity_E_array.sum / REMOLD_alpha_res.length
//    val REMOLD_purity_var = REMOLD_purity_E2_array.sum / REMOLD_alpha_res.length - pow(REMOLD_purity_mean, 2)
//    println("purity mean = " + REMOLD_purity_mean + "  purity var = " + REMOLD_purity_var)
//    println("purity mean top 20% = " + REMOLD_all_purity_res.sum / REMOLD_all_purity_res.length + "   purity var top 20% = " + (REMOLD_all_purity_res.map(x => x * x).sum / REMOLD_all_purity_res.length - pow(REMOLD_all_purity_res.sum / REMOLD_all_purity_res.length, 2)))
//
////    val REMOLD_writer_model = new PrintWriter(new File("model/SVD_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_model.txt"))
////    for (model <- REMOLD_localGuassianModel) {
////      REMOLD_writer_model.write(model.partitionId.toString + '\t' + model.modelId.toString + '\t' + model.datasize.toString + '\t' + model.mean.toArray.mkString(",") + '\t' + model.inverseMatrix.toArray.mkString(",") + '\t' + model.logSqrtDet.toString + "\n")
////    }
////    REMOLD_writer_model.close()
////
////    val REMOLD_writer_cluster_res_nmi = new PrintWriter(new File("cluster_res/SVD_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_cluster_res_nmi.txt"))
////    for (res_line <- REMOLD_best_cluster_res_nmi) {
////      REMOLD_writer_cluster_res_nmi.write(res_line + "\n")
////    }
////    REMOLD_writer_cluster_res_nmi.close()
////
////    val REMOLD_writer_cluster_res_purity = new PrintWriter(new File("cluster_res/SVD_" + svd_ratio + "_" + datasetName + "_" + hash_seed + "_cluster_res_purity.txt"))
////    for (res_line <- REMOLD_best_cluster_res_purity) {
////      REMOLD_writer_cluster_res_purity.write(res_line + "\n")
////    }
////    REMOLD_writer_cluster_res_purity.close()

    sc.stop()
    sys.exit(0)

  }
}
