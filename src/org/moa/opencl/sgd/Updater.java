/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.moa.opencl.sgd;

import org.viennacl.binding.Buffer;

/**
 *
 * @author bsp
 */
public interface Updater {

  void applyUpdate(Buffer gradient_buffer, int batch_number);

  double[] getBias();

  double[] getWeights();

  void readWeights(Buffer weights);
  
}
