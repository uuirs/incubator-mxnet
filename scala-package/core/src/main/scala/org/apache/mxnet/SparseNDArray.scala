/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet

import java.nio.{ByteBuffer, ByteOrder}

import org.apache.mxnet
import org.apache.mxnet.SType
import org.apache.mxnet.SType.SType
import org.apache.mxnet.Base._
import org.apache.mxnet.DType.DType
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.ref.WeakReference

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet. <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * </b>
 */
object BaseSparseNDArray {

  val STORAGE_AUX_TYPES = Map(SType.ROW_SPARSE -> List(DType.Int64), SType.CSR -> List(DType.Int64,DType.Int64))

  private def newEmptyHandle(): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl.value
  }

  private def newAllocHandle(stype: SType, shape: Shape, ctx: Context, delayAlloc: Boolean, dtype: DType,
                     auxTypes: List[DType], auxShapes: List[Shape] = null): NDArrayHandle = {
    val hdl = new NDArrayHandleRef

    val auxShapesList = if (auxShapes == null) auxTypes.map(_ => Shape(0)) else auxShapes

    checkCall(_LIB.mxNDArrayCreateSparseEx(
      stype.id,
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      dtype.id,
      auxTypes.size,
      auxTypes.toArray.map(_.id),
      auxShapesList.toArray.map(_.length),
      auxShapesList.toArray.map(_.toArray).flatten,
      hdl))
    hdl.value
  }

  def empty(stype: SType, shape: Shape, ctx: Context, delayAlloc: Boolean = false, dtype: DType = DType.Float32,
            auxTypes: List[DType], auxShape: List[Shape] = null): NDArray = {
    new NDArray(handle = newAllocHandle(stype, shape, ctx, delayAlloc, dtype,
      auxTypes, auxShape))
  }

}

class BaseSparseNDArray private[mxnet](private[mxnet] override val handle: NDArrayHandle,
                                       override val writable: Boolean = true,
                             addToCollector: Boolean = true) extends NDArray(handle, writable, addToCollector) with WarnIfNotDisposed {

  // record arrays who construct this array instance
  // we use weak reference to prevent gc blocking
  @volatile private var disposed = false

  /**
   * Peform an synchronize copy from the array.
   * @param source The data source we should like to copy from.
   */
  private def syncCopyfrom(source: Array[Float]): Unit = {
    throw new UnsupportedOperationException("not supported for SparseNDArray")
  }


  /**
   * Return a sub NDArray that shares memory with current one.
   * the first axis will be rolled up, which causes its shape different from slice(i, i+1)
   * @param idx index of sub array.
   */
  override def at(idx: Int): NDArray = {
    throw new UnsupportedOperationException("not supported for SparseNDArray")
  }

  // Get transpose of current NDArray
  override def T: NDArray = {
    throw new UnsupportedOperationException("not supported for SparseNDArray")
  }


  /**
   * Return a copied numpy array of current array with specified type.
   * @param dtype Desired type of result array.
   * @return A copy of array content.
   */

  override def asType(dtype: DType): NDArray = {
    throw new UnsupportedOperationException("not supported for SparseNDArray")
  }

  def auxType(i: Int): DType = {
    val mxAuxType = new RefInt
    checkCall(_LIB.mxNDArrayGetAuxType(handle, i, mxAuxType))
    DType(mxAuxType.value)
  }

  def numAux : Int = BaseSparseNDArray.STORAGE_AUX_TYPES(this.stype).size

  def auxTypes: List[DType] = {
    val auxTypeList = ListBuffer.empty[DType]
    for ( i <- 0 until numAux) {
      auxTypeList += auxType(i)
    }
    auxTypeList.toList
  }

  def data: NDArray = {
    waitToRead()
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayGetDataNDArray(this.handle, hdl))
    new NDArray(hdl.value)
  }

  def auxData(i: Int): NDArray = {
    waitToRead()
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayGetAuxNDArray(this.handle, i, hdl))
    new NDArray(hdl.value)
  }

  /**
   * Return a reshaped NDArray that shares memory with current one.
   * @param dims New shape.
   *
   * @return a reshaped NDArray that shares memory with current one.
   */
  override def reshape(dims: Array[Int]): NDArray = {
    throw new UnsupportedOperationException()
  }

  /**
   * Return a reshaped NDArray that shares memory with current one.
   * @param dims New shape.
   *
   * @return a reshaped NDArray that shares memory with current one.
   */
  override def reshape(dims: Shape): NDArray = {
    throw new UnsupportedOperationException()
  }

  /**
   * Set the values of the NDArray
   * @param value Value to set
   * @return Current NDArray
   */
  override def set(value: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def set(other: Array[Float]): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def +(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()  }

  override def +(other: Float): NDArray = {
    throw new UnsupportedOperationException()  }

  override def +=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def +=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def -(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()  }

  override def -(other: Float): NDArray = {
    throw new UnsupportedOperationException()  }

  override def -=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def -=(other: Float): NDArray = {
    throw new UnsupportedOperationException()

  }

  override def *(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def *(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def unary_-(): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def *=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def *=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def /(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def /(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def /=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def /=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def **(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def **(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def **=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def **=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def >(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def >(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def >=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def >=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def <(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def <(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def <=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def <=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def %(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def %(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def %=(other: NDArray): NDArray = {
    throw new UnsupportedOperationException()
  }

  override def %=(other: Float): NDArray = {
    throw new UnsupportedOperationException()
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * @return  A copy of array content.
   */
  override def toArray: Array[Float] = {
    throw new UnsupportedOperationException()
  }

  override def internal: NDArrayInternal = {
    throw new UnsupportedOperationException()
    val myType = dtype
    val arrLength = DType.numOfBytes(myType) * size
    val arr = Array.ofDim[Byte](arrLength)
    checkCall(_LIB.mxNDArraySyncCopyToCPU(handle, arr, size))
    new NDArrayInternal(arr, myType)
  }

  /**
   * Return a CPU scalar(float) of current ndarray.
   * This ndarray must have shape (1,)
   *
   * @return The scalar representation of the ndarray.
   */
  override def toScalar: Float = {
    throw new UnsupportedOperationException()
    require(shape == Shape(1), "The current array is not a scalar")
    this.toArray(0)
  }

  /**
   * Copy the content of current array to a new NDArray in the context.
   *
   * @param ctx Target context we want to copy data to.
   * @return The copy target NDArray
   */
  override def copyTo(ctx: Context): NDArray = {
    val ret = BaseSparseNDArray.empty(stype, shape, ctx, delayAlloc = true, dtype, auxTypes)
    copyTo(ret)
  }

  // Get size of current NDArray.
  override def size: Int = shape.product


  override def equals(o: Any): Boolean = o match {
    case that: NDArray =>
      that != null && that.shape == this.shape && that.toArray.sameElements(this.toArray)
    case _ => false
  }

  override def hashCode: Int = {
    // TODO: naive implementation
    shape.hashCode + toArray.hashCode
  }

}
object CSRNDArray {
  def CSRNDArray(data: Array[Float], indices: Array[Int], indptr: Array[Int],
                 shape: Shape = null, ctx: Context = null,
                 dtype: DType = DType.Float32): NDArray = {
    val storageType = SType.CSR
    val ndData = NDArray.array(data, Shape(data.size))
    val ndIndices = NDArray.array(indices.map(_.toLong), Shape(indices.size), ctx)
    val ndIndptr = NDArray.array(indptr.map(_.toLong), Shape(indptr.size), ctx)

    val indptrType = DType.Int64
    val indiceType = DType.Int64

    val context = if (ctx == null) Context.defaultCtx else ctx

    val  dataShape = if (shape != null) shape else
      Shape(ndIndptr.size - 1, NDArray.api.max(ndIndices).get.internal.toLongArray(0).toInt  + 1)

    val auxShapes = List(ndIndices.shape, ndIndptr.shape)

    val result = BaseSparseNDArray.empty(storageType, dataShape, context, false, dtype,
      List(indptrType, indiceType), auxShapes)

    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(result.handle, ndData.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(result.handle, ndIndptr.handle, 0))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(result.handle, ndIndices.handle, 1))
    result
  }
}

class CSRNDArray private[mxnet](private[mxnet] override val handle: NDArrayHandle,
                                                        override val writable: Boolean = true,
                                                        addToCollector: Boolean = true) extends BaseSparseNDArray(handle, writable, addToCollector) with WarnIfNotDisposed {

  def indices = auxData(1)
  def indptr = auxData(0)
  //def data: NDArray


}

object RowSparseNDArray {
  def RowSparseNDArray(data: Array[Float], indices: Array[Int],
                 shape: Shape = null, ctx: Context = null,
                 dtype: DType = null): NDArray = {
    val storageType = SType.ROW_SPARSE
    val ndData = NDArray.array(data, Shape(data.size))
    val ndIndices = NDArray.array(indices.map(_.toLong), Shape(indices.size), ctx)

    val indiceType = DType.Int64

    val context = if (ctx == null) Context.defaultCtx else ctx

    val  dataShape = if (shape != null) shape else
      Shape(indices(indices.length - 1) + 1, 1)

    val result = BaseSparseNDArray.empty(storageType, dataShape, context, false, dtype,
      List(indiceType), List(ndIndices.shape))

    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(result.handle, ndData.handle, -1))
    checkCall(_LIB.mxNDArraySyncCopyFromNDArray(result.handle, ndIndices.handle, 0))
    result
  }
}

class RowSparseNDArray private[mxnet](private[mxnet] override val handle: NDArrayHandle,
                                      override val writable: Boolean = true,
                                      addToCollector: Boolean = true) extends BaseSparseNDArray(handle, writable, addToCollector) with WarnIfNotDisposed {

  def indices = auxData(0)
  //def data: NDArray


}

