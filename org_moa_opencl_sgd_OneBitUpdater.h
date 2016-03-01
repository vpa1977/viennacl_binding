/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_opencl_sgd_OneBitUpdater */

#ifndef _Included_org_moa_opencl_sgd_OneBitUpdater
#define _Included_org_moa_opencl_sgd_OneBitUpdater
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_moa_opencl_sgd_OneBitUpdater
 * Method:    nativeComputeUpdate
 * Signature: (JJJJJJIIDI)V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_sgd_OneBitUpdater_nativeComputeUpdate
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jlong, jlong, jint, jint, jdouble, jint);

/*
 * Class:     org_moa_opencl_sgd_OneBitUpdater
 * Method:    nativeComputeWeight
 * Signature: (JJJDII)V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_sgd_OneBitUpdater_nativeComputeWeight
  (JNIEnv *, jobject, jlong, jlong, jlong, jdouble, jint, jint);

/*
 * Class:     org_moa_opencl_sgd_OneBitUpdater
 * Method:    nativeUpdateTau
 * Signature: (JJJJJIIII)V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_sgd_OneBitUpdater_nativeUpdateTau
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jlong, jint, jint, jint, jint);

#ifdef __cplusplus
}
#endif
#endif