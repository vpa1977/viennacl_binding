/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_moa_opencl_util_DoubleBitonicSort */

#ifndef _Included_org_moa_opencl_util_DoubleBitonicSort
#define _Included_org_moa_opencl_util_DoubleBitonicSort
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_moa_opencl_util_DoubleBitonicSort
 * Method:    init
 * Signature: (Lorg/viennacl/binding/Context;)V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_util_DoubleBitonicSort_init
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_moa_opencl_util_DoubleBitonicSort
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_util_DoubleBitonicSort_release
  (JNIEnv *, jobject);

/*
 * Class:     org_moa_opencl_util_DoubleBitonicSort
 * Method:    nativeSort
 * Signature: (Lorg/viennacl/binding/Buffer;Lorg/viennacl/binding/Buffer;Lorg/viennacl/binding/Buffer;Lorg/viennacl/binding/Buffer;I)V
 */
JNIEXPORT void JNICALL Java_org_moa_opencl_util_DoubleBitonicSort_nativeSort
  (JNIEnv *, jobject, jobject, jobject, jobject, jobject, jint);

#ifdef __cplusplus
}
#endif
#endif
