/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_viennacl_binding_Context */

#ifndef _Included_org_viennacl_binding_Context
#define _Included_org_viennacl_binding_Context
#ifdef __cplusplus
extern "C" {
#endif
#undef org_viennacl_binding_Context_MAIN_MEMORY
#define org_viennacl_binding_Context_MAIN_MEMORY 0L
#undef org_viennacl_binding_Context_OPENCL_MEMORY
#define org_viennacl_binding_Context_OPENCL_MEMORY 1L
#undef org_viennacl_binding_Context_HSA_MEMORY
#define org_viennacl_binding_Context_HSA_MEMORY 2L
/*
 * Class:     org_viennacl_binding_Context
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Context_release
  (JNIEnv *, jobject);

/*
 * Class:     org_viennacl_binding_Context
 * Method:    init
 * Signature: (ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Context_init
  (JNIEnv *, jobject, jint, jstring);

/*
 * Class:     org_viennacl_binding_Context
 * Method:    addProgram
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Context_addProgram
  (JNIEnv *, jobject, jstring, jstring);

/*
 * Class:     org_viennacl_binding_Context
 * Method:    nativeGetKernel
 * Signature: (Ljava/lang/String;Ljava/lang/String;Lorg/viennacl/binding/Kernel;)Lorg/viennacl/binding/Kernel;
 */
JNIEXPORT jobject JNICALL Java_org_viennacl_binding_Context_nativeGetKernel
  (JNIEnv *, jobject, jstring, jstring, jobject);

/*
 * Class:     org_viennacl_binding_Context
 * Method:    removeProgram
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_viennacl_binding_Context_removeProgram
  (JNIEnv *, jobject, jstring);

/*
 * Class:     org_viennacl_binding_Context
 * Method:    createQueue
 * Signature: ()Lorg/viennacl/binding/Queue;
 */
JNIEXPORT jobject JNICALL Java_org_viennacl_binding_Context_createQueue
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
