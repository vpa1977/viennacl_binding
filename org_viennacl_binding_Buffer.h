/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_viennacl_binding_Buffer */

#ifndef _Included_org_viennacl_binding_Buffer
#define _Included_org_viennacl_binding_Buffer
#ifdef __cplusplus
extern "C" {
#endif
#undef org_viennacl_binding_Buffer_WRITE
#define org_viennacl_binding_Buffer_WRITE 1L
#undef org_viennacl_binding_Buffer_READ
#define org_viennacl_binding_Buffer_READ 2L
#undef org_viennacl_binding_Buffer_READ_WRITE
#define org_viennacl_binding_Buffer_READ_WRITE 3L
/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    fill
 * Signature: (B)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_fill__B
  (JNIEnv *, jobject, jbyte);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    fill
 * Signature: (BJ)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_fill__BJ
  (JNIEnv *, jobject, jbyte, jlong);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    map
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_map__I
  (JNIEnv *, jobject, jint);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    map
 * Signature: (IJJ)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_map__IJJ
  (JNIEnv *, jobject, jint, jlong, jlong);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    commit
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_commit
  (JNIEnv *, jobject);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    native_copy
 * Signature: (Lorg/viennacl/binding/Buffer;)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_native_1copy__Lorg_viennacl_binding_Buffer_2
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    native_copy
 * Signature: (Lorg/viennacl/binding/Buffer;JJ)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_native_1copy__Lorg_viennacl_binding_Buffer_2JJ
  (JNIEnv *, jobject, jobject, jlong, jlong);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    allocate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_allocate
  (JNIEnv *, jobject);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_release__
  (JNIEnv *, jobject);

/*
 * Class:     org_viennacl_binding_Buffer
 * Method:    release
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Buffer_release__J
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
