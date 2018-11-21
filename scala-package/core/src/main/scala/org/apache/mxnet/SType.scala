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

object SType extends Enumeration {
  type SType = Value
  val UNDEFINED = Value(-1, "undefined")
  val DEFAULT = Value(0, "default")
  val ROW_SPARSE = Value(1, "row_sparse")
  val CSR = Value(2, "csr")

  private[mxnet] def getSType(stypeStr: String): SType = {
    stypeStr match {
      case "undefined" => SType.UNDEFINED
      case "default" => SType.DEFAULT
      case "row_sparse" => SType.ROW_SPARSE
      case "csr" => SType.CSR
      case _ => throw new IllegalArgumentException(
        s"SType: $stypeStr not found! please set it in SType.scala")
    }
  }
}

