
set PATH=%PATH%;"C:\Program Files\Java\jdk1.8.0_45\bin"
set CLASSPATH=C:\master_in_progress\java\viennacl-binding\target\classes;C:\master_in_progress\hsa_jni\viennacl-frontend\moa-frontend\target\classes;C:\Users\john\.m2\repository\com\typesafe\akka\akka-actor_2.10\2.3.9\akka-actor_2.10-2.3.9.jar;C:\Users\john\.m2\repository\org\scala-lang\scala-library\2.10.4\scala-library-2.10.4.jar;C:\Users\john\.m2\repository\com\typesafe\config\1.2.1\config-1.2.1.jar;C:\Users\john\.m2\repository\com\typesafe\akka\akka-stream-experimental_2.11\1.0-M2\akka-stream-experimental_2.11-1.0-M2.jar;C:\Users\john\.m2\repository\com\typesafe\akka\akka-actor_2.11\2.3.7\akka-actor_2.11-2.3.7.jar;C:\Users\john\.m2\repository\org\reactivestreams\reactive-streams\1.0.0.M3\reactive-streams-1.0.0.M3.jar;C:\Users\john\.m2\repository\junit\junit\4.4\junit-4.4.jar;C:\Users\john\.m2\repository\nz\ac\waikato\cms\weka\weka-dev\3.7.10\weka-dev-3.7.10.jar;C:\Users\john\.m2\repository\net\sf\squirrel-sql\thirdparty-non-maven\java-cup\0.11a\java-cup-0.11a.jar;C:\Users\john\.m2\repository\org\pentaho\pentaho-commons\pentaho-package-manager\1.0.8\pentaho-package-manager-1.0.8.jar;C:\Users\john\.m2\repository\com\nativelibs4java\javacl\1.0.0-RC3\javacl-1.0.0-RC3.jar;C:\Users\john\.m2\repository\com\nativelibs4java\javacl-core\1.0.0-RC3\javacl-core-1.0.0-RC3.jar;C:\Users\john\.m2\repository\com\nativelibs4java\opencl4java\1.0.0-RC3\opencl4java-1.0.0-RC3.jar;C:\Users\john\.m2\repository\com\nativelibs4java\bridj\0.6.2\bridj-0.6.2.jar;C:\Users\john\.m2\repository\com\google\android\tools\dx\1.7\dx-1.7.jar;C:\Users\john\.m2\repository\com\nativelibs4java\nativelibs4java-utils\1.5\nativelibs4java-utils-1.5.jar;C:\Users\john\.m2\repository\nz\ac\waikato\cms\moa\moa\2013.11\moa-2013.11.jar;C:\Users\john\.m2\repository\com\googlecode\sizeofag\sizeofag\1.0.0\sizeofag-1.0.0.jar


javah -cp %CLASSPATH% org.viennacl.binding.Context
javah -cp %CLASSPATH% org.viennacl.binding.Kernel
javah -cp %CLASSPATH% org.viennacl.binding.DoubleVector
javah -cp %CLASSPATH% org.viennacl.binding.Matrix
javah -cp %CLASSPATH% org.viennacl.binding.Operations
javah -cp %CLASSPATH% org.viennacl.binding.Buffer
javah -cp %CLASSPATH% org.viennacl.binding.Queue
javah -cp %CLASSPATH% org.moa.opencl.util.DoubleMergeSort
javah -cp %CLASSPATH% org.moa.opencl.util.FloatMergeSort
javah -cp %CLASSPATH% org.moa.opencl.util.MortonCode
javah -cp %CLASSPATH% org.moa.opencl.util.CLogsSort

