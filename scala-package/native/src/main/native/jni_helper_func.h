/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file jni_helper_func.h
 * \brief Helper functions for operating JVM objects
 */
#include <jni.h>

#ifndef MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
#define MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_

jlong GetLongField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("org/apache/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  jlong ret = env->GetLongField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

jint GetIntField(JNIEnv *env, jobject obj) {
  jclass refClass = env->FindClass("org/apache/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  jint ret = env->GetIntField(obj, refFid);
  env->DeleteLocalRef(refClass);
  return ret;
}

void SetIntField(JNIEnv *env, jobject obj, jint value) {
  jclass refClass = env->FindClass("org/apache/mxnet/Base$RefInt");
  jfieldID refFid = env->GetFieldID(refClass, "value", "I");
  env->SetIntField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void SetLongField(JNIEnv *env, jobject obj, jlong value) {
  jclass refClass = env->FindClass("org/apache/mxnet/Base$RefLong");
  jfieldID refFid = env->GetFieldID(refClass, "value", "J");
  env->SetLongField(obj, refFid, value);
  env->DeleteLocalRef(refClass);
}

void SetStringField(JNIEnv *env, jobject obj, const char *value) {
  jclass refClass = env->FindClass("org/apache/mxnet/Base$RefString");
  jfieldID refFid = env->GetFieldID(refClass, "value", "Ljava/lang/String;");
  env->SetObjectField(obj, refFid, env->NewStringUTF(value));
  env->DeleteLocalRef(refClass);
}

const char ** CreateStringArrayField(JNIEnv *env, jobjectArray objArr, size_t size) {

  const char **strArr = new const char *[size];
  for (size_t i = 0; i < size; i++) {
    jstring jkey = reinterpret_cast<jstring>(env->GetObjectArrayElement(objArr, i));
    const char *key = env->GetStringUTFChars(jkey, 0);
    strArr[i] = key;
    env->DeleteLocalRef(jkey);
  }
  return strArr;
}

void ReleaseStringArrayField(JNIEnv *env, jobjectArray objArr, const char **strArr, size_t size) {
  for (size_t i = 0; i < size; i++) {
    jstring jkey = (jstring) env->GetObjectArrayElement(objArr, i);
    env->ReleaseStringUTFChars(jkey, strArr[i]);
    env->DeleteLocalRef(jkey);
  }
  delete[] strArr;
}

void SetStringArrayField(JNIEnv *env, jobject obj, const char * const *strArr, size_t size) {
  jclass arrayClass = env->FindClass("scala/collection/mutable/ArrayBuffer");
  jmethodID arrayAppend = env->GetMethodID(arrayClass,
    "$plus$eq", "(Ljava/lang/Object;)Lscala/collection/mutable/ArrayBuffer;");
  for (size_t i = 0; i < size; ++i) {
    jstring jtmp = env->NewStringUTF(strArr[i]);
    env->CallObjectMethod(obj, arrayAppend, jtmp);
    env->DeleteLocalRef(jtmp);
  }
}

void SetLongArrayField(JNIEnv *env, jobject obj, const uint64_t *longArr, size_t size) {
  jclass arrayClass = env->FindClass("scala/collection/mutable/ArrayBuffer");
  jmethodID arrayAppend = env->GetMethodID(arrayClass,
    "$plus$eq", "(Ljava/lang/Object;)Lscala/collection/mutable/ArrayBuffer;");
  jclass longCls = env->FindClass("java/lang/Long");
  jmethodID longConst = env->GetMethodID(longCls, "<init>", "(J)V");

  for (size_t i = 0; i < size; ++i) {
    jobject handle = env->NewObject(longCls, longConst, longArr[i]);
    env->CallObjectMethod(obj, arrayAppend, handle);
    env->DeleteLocalRef(handle);
  }
}

#endif  // MXNET_JNICPP_MAIN_NATIVE_JNI_HELPER_FUNC_H_
